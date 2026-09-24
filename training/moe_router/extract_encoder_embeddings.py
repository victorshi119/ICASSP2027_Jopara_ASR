#!/usr/bin/env python3
"""
M1 Step 1 — Extract pooled encoder embeddings from the frozen OmniASR encoder.

The wav2vec2 encoder is identical for all expert branches (LoRA only modifies the
decoder). Running audio through the encoder once gives acoustic embeddings that
capture regime-relevant phonetic and prosodic information.

We hook into the encoder→decoder projection (encoder_proj) to capture the encoder
output just before it enters the LLM decoder. This gives a [T, D] tensor per utterance;
we mean-pool to [D] for the classifier.

Outputs (in results/June_14_Multitask_Audio_LLM/embeddings/):
  train_embeddings.npy    shape [N_train, D]
  train_labels.npy        string array [N_train]  (spa / cs / grn)
  train_utt_ids.json      list of utt_ids in same order
  dev_embeddings.npy      shape [N_dev, D]    ← used for hyperparameter selection
  dev_labels.npy          string array [N_dev]
  dev_utt_ids.json        list of utt_ids
  test_embeddings.npy     shape [150, D]      ← final eval only
  test_labels.npy         string array [150]
  test_utt_ids.json       list of utt_ids

DO NOT extract or tune on test embeddings until all hyperparameters are frozen.

Usage:
  cd /media/volume/sapc2_data/victor/icassp2026/Jopara_ASR_models
  source env.sh
  $OMNI_PY results/June_14_Multitask_Audio_LLM/scripts/extract_encoder_embeddings.py --split train
  $OMNI_PY results/June_14_Multitask_Audio_LLM/scripts/extract_encoder_embeddings.py --split dev
  $OMNI_PY results/June_14_Multitask_Audio_LLM/scripts/extract_encoder_embeddings.py --split test  # only once frozen
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

BASE          = Path("/media/volume/sapc2_data/victor/icassp2026/Jopara_ASR_models")
MANIFEST_DIR  = BASE / "results/June_14_Multitask_Audio_LLM/manifests"
OUT_DIR       = BASE / "results/June_14_Multitask_Audio_LLM/embeddings"
TEST_MANIFEST = BASE / "moe_lora/manifests/v2_test_moe_manifest.jsonl"
DEV_MANIFEST  = BASE / "moe_lora/manifests/v3_dev_moe_manifest.jsonl"

REGIME_MAP = {"spanish_only": "spa", "mixed": "cs", "jehe'a": "cs", "guarani_only": "grn"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_rows(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]


class EncoderHook:
    """Captures the output of the encoder→decoder projection layer."""

    def __init__(self):
        self.outputs: list[torch.Tensor] = []
        self._handle = None

    def register(self, model: torch.nn.Module) -> str:
        candidates = []
        for name, mod in model.named_modules():
            n = name.lower()
            if "encoder_proj" in n:
                candidates.insert(0, (name, mod, "encoder_proj"))
            elif "encoder_to_decoder" in n or "enc2dec" in n:
                candidates.insert(0, (name, mod, "enc2dec"))
            elif n.startswith("encoder.") and isinstance(mod, torch.nn.Linear):
                candidates.append((name, mod, "encoder_linear"))

        if not candidates:
            raise RuntimeError(
                "Could not find encoder projection layer. "
                "Run list_modules.py to inspect model structure."
            )

        name, mod, kind = candidates[0]
        log.info("Hooking into layer '%s' (type: %s, kind: %s)", name, type(mod).__name__, kind)
        self._handle = mod.register_forward_hook(self._hook_fn)
        return name

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if out.dim() == 3:
            pooled = out.mean(dim=1)
        elif out.dim() == 2:
            pooled = out.mean(dim=0, keepdim=True)
        else:
            pooled = out.unsqueeze(0)
        self.outputs.append(pooled.detach().cpu().float())

    def pop_all(self) -> torch.Tensor:
        out = torch.cat(self.outputs, dim=0)
        self.outputs.clear()
        return out

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


def extract_split(
    rows: list[dict],
    pipeline,
    hook: EncoderHook,
    batch_size: int,
    checkpoint_prefix: str | None = None,
    checkpoint_every: int = 200,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract embeddings with optional incremental checkpointing.

    If checkpoint_prefix is set (e.g. 'train'), saves partial results every
    checkpoint_every utterances so a crash doesn't lose all progress.
    On restart, pass the same rows — already-done rows are skipped via the
    checkpoint file.
    """
    import omnilingual_asr.models.inference.pipeline as _pipe
    _pipe.MAX_ALLOWED_AUDIO_SEC = float("inf")

    # ── Resume from checkpoint if available ──────────────────────────────────
    ckpt_path = OUT_DIR / f"{checkpoint_prefix}_ckpt.npz" if checkpoint_prefix else None
    start_idx = 0
    all_embeds: list[torch.Tensor] = []
    labels: list[str] = []
    utt_ids: list[str] = []

    if ckpt_path and ckpt_path.exists():
        ckpt = np.load(ckpt_path, allow_pickle=True)
        prev_embeds = ckpt["embeddings"]
        prev_labels = ckpt["labels"].tolist()
        prev_utt_ids = ckpt["utt_ids"].tolist()
        done_set = set(prev_utt_ids)
        # Find how many of our rows are already done
        start_idx = sum(1 for r in rows if r["utt_id"] in done_set)
        all_embeds.append(torch.from_numpy(prev_embeds))
        labels.extend(prev_labels)
        utt_ids.extend(prev_utt_ids)
        log.info("Resumed from checkpoint: %d/%d done", start_idx, len(rows))

    t0 = time.time()
    for i in range(start_idx, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        audio_paths = [r["audio_path"] for r in batch]
        lang_tags = ["grn_Latn"] * len(audio_paths)  # encoder is unconditional

        hook.outputs.clear()
        try:
            _ = pipeline.transcribe(audio_paths, lang=lang_tags)
        except Exception as e:
            log.warning("Batch %d–%d failed: %s — skipping", i, i+len(batch), e)
            # Insert zero embeddings so indices stay aligned
            dummy = torch.zeros(len(batch), 4096)
            all_embeds.append(dummy)
            for r in batch:
                labels.append("unknown")
                utt_ids.append(r["utt_id"])
            continue

        batch_embeds = hook.pop_all()

        if batch_embeds.shape[0] != len(batch):
            log.warning("Hook captured %d tensors for batch of %d", batch_embeds.shape[0], len(batch))
            if batch_embeds.shape[0] > len(batch):
                batch_embeds = batch_embeds[:len(batch)]
            else:
                pad = torch.zeros(len(batch) - batch_embeds.shape[0], batch_embeds.shape[1])
                batch_embeds = torch.cat([batch_embeds, pad], dim=0)

        all_embeds.append(batch_embeds)
        for r in batch:
            labels.append(r.get("regime", REGIME_MAP.get(r.get("v2_category", ""), "unknown")))
            utt_ids.append(r["utt_id"])

        done = min(i + batch_size, len(rows))
        if done % 100 == 0 or done == len(rows):
            elapsed = time.time() - t0
            log.info("  %d/%d utterances done (%.0fs, %.2fs/utt)",
                     done, len(rows), elapsed, elapsed / (done - start_idx) if done > start_idx else 0)

        # Save checkpoint
        if ckpt_path and done % checkpoint_every == 0:
            merged = torch.cat(all_embeds, dim=0).numpy()
            np.savez(ckpt_path, embeddings=merged,
                     labels=np.array(labels), utt_ids=np.array(utt_ids))
            log.info("  Checkpoint saved at %d utterances → %s", done, ckpt_path.name)

    embeddings = torch.cat(all_embeds, dim=0).numpy()
    return embeddings, labels, utt_ids


def save_embeddings(split: str, embeds: np.ndarray, labels: list[str], utt_ids: list[str]):
    np.save(OUT_DIR / f"{split}_embeddings.npy", embeds)
    np.save(OUT_DIR / f"{split}_labels.npy", np.array(labels))
    (OUT_DIR / f"{split}_utt_ids.json").write_text(json.dumps(utt_ids))
    log.info("%s embeddings saved: shape %s", split, embeds.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True,
                        help="Which split to extract. Extract train and dev first; "
                             "test only after hyperparameters are frozen.")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from fairseq2.models.hub import load_model
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from omnilingual_asr.models.inference import ASRInferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading base model on %s (float16) ...", device)
    t0 = time.time()
    model = load_model("omniASR_LLM_7B_v2", device=torch.device("cpu"), dtype=torch.float16)
    tokenizer = load_tokenizer("omniASR_LLM_7B_v2")
    model.to(device=torch.device(device), dtype=torch.float16)
    log.info("Model loaded in %.1fs (%.1f GB VRAM)",
             time.time() - t0,
             torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0)

    hook = EncoderHook()
    hooked_layer = hook.register(model)
    log.info("Encoder hook registered on: %s", hooked_layer)

    pipeline = ASRInferencePipeline(
        model_card=None, model=model, tokenizer=tokenizer,
        device=device, dtype=torch.float16,
    )

    if args.split == "test":
        log.info("=== TEST SPLIT (150 utterances) — final eval only ===")
        rows = load_rows(TEST_MANIFEST)
        for r in rows:
            r["regime"] = REGIME_MAP.get(r.get("v2_category", ""), "unknown")
        embeds, labels, utt_ids = extract_split(rows, pipeline, hook, args.batch_size)
        save_embeddings("test", embeds, labels, utt_ids)

    elif args.split == "dev":
        # Dev = v3_dev_moe_manifest (CEGPA spa+cs, 217 utt)
        #       + dev_grn.jsonl (CV+CEGPA grn dev, 286 utt)
        # This gives full 3-class dev coverage.
        log.info("=== DEV SPLIT ===")
        dev_rows = load_rows(DEV_MANIFEST)
        for r in dev_rows:
            r["regime"] = REGIME_MAP.get(r.get("v2_category", ""), "unknown")
            # normalise reference field name
            if "reference_norm" in r and "reference" not in r:
                r["reference"] = r["reference_norm"]

        # Add Guaraní dev (from manifests built by build_train_manifest.py)
        grn_dev_path = BASE / "results/June_14_Multitask_Audio_LLM/manifests/dev_grn.jsonl"
        if grn_dev_path.exists():
            grn_rows = load_rows(grn_dev_path)
            log.info("CEGPA+CV dev (spa+cs): %d  |  grn dev: %d", len(dev_rows), len(grn_rows))
            dev_rows = dev_rows + grn_rows
        else:
            log.warning("dev_grn.jsonl not found — dev set will lack Guaraní. "
                        "Run build_train_manifest.py first.")

        from collections import Counter
        log.info("Dev regime distribution: %s", dict(Counter(r["regime"] for r in dev_rows)))
        embeds, labels, utt_ids = extract_split(dev_rows, pipeline, hook, args.batch_size)
        save_embeddings("dev", embeds, labels, utt_ids)

    elif args.split == "train":
        train_manifest = MANIFEST_DIR / "train_all.jsonl"
        if not train_manifest.exists():
            log.error("train_all.jsonl not found — run build_train_manifest.py first")
            return
        log.info("=== TRAIN SPLIT ===")
        rows = load_rows(train_manifest)
        log.info("Total training rows: %d", len(rows))
        embeds, labels, utt_ids = extract_split(
            rows, pipeline, hook, args.batch_size,
            checkpoint_prefix="train", checkpoint_every=200,
        )
        save_embeddings("train", embeds, labels, utt_ids)
        # Clean up checkpoint after successful save
        ckpt = OUT_DIR / "train_ckpt.npz"
        if ckpt.exists():
            ckpt.unlink()
            log.info("Checkpoint removed after successful save")

    hook.remove()
    log.info("Done. Embeddings in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
