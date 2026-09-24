#!/usr/bin/env python3
"""
Stage A4 — Run J-MoLE-ASR inference and generate hypothesis packs.

Produces per-utterance JSON with:
  - 1-best from the full J-MoLE model (soft-mixed experts)
  - pseudo-N-best from each individual expert (one-hot gate)
  - predicted regime probabilities [p_spa, p_cs, p_grn]

The output is used directly by build_hypothesis_pack.py to build GER inputs.

Usage:
  # A2 checkpoint (router only):
  $OMNI_PY scripts/infer_jmole_asr.py --config configs/jmole_grn.yaml --split test

  # A3 checkpoint (joint router + LoRA, best by default):
  $OMNI_PY scripts/infer_jmole_asr.py --config configs/jmole_grn.yaml --split test \\
      --a3-ckpt checkpoints/jmole_grn_a3/step_1200_best.pt

  # With a Jopara in-context prefix (biases LLaMA decoder toward CS vocabulary):
  $OMNI_PY scripts/infer_jmole_asr.py --config configs/jmole_grn.yaml --split test \\
      --a3-ckpt checkpoints/jmole_grn_a3/step_1200_best.pt --jopara-prefix

  # Quick sanity check (first 10 utterances, no per-expert hypotheses):
  $OMNI_PY scripts/infer_jmole_asr.py --config configs/jmole_grn.yaml --split test \\
      --limit 10 --no-expert-hyps

Notes on Jopara prefix (--jopara-prefix):
  OmniASR's standard transcribe() does not expose a text-prefix API for the
  LLM_ASR_LID model variant (transcribe_with_context() is only for ZERO_SHOT).
  We implement prefix injection manually: after model(batch, return_decoder_inputs=True)
  returns the audio-conditioned decoder context, we embed a short Jopara phrase
  via the LLaMA embedding layer and concatenate it onto the context before passing
  to generate_hypotheses_one_segment(). This primes the beam-search decoder toward
  code-switching vocabulary without any gradient updates.

  Default prefix: a short Guarani-Spanish Jopara phrase drawn from CEGPA style.
  Can be overridden with --prefix-text "your text here".
"""
from __future__ import annotations

import os
import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJ_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))  # data root (original project layout)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "lora_adapters"))

import torch
import yaml

from model.moe_lora_linear import (
    GateStore, EXPERT_KEYS, MoELoRALinear,
)
from model.patch_omniasr_decoder import (
    build_jmole_model,
    set_onehot_gate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE  = _PROJ_ROOT
J14   = BASE / "results/June_14_Multitask_Audio_LLM"
J15   = BASE / "results/June_15_Multitask_Audio_LLM"

MANIFEST_MAP = {
    "train": J14 / "manifests/train_all.jsonl",
    "dev":   J14 / "manifests/dev_cs.jsonl",
    "test":  BASE / "moe_lora/manifests/v2_test_moe_manifest.jsonl",
}

EXPERT_NAMES = {
    "spa": "E_spa (Spanish expert)",
    "cs":  "E_cs  (CS-Jopara expert)",
    "grn": "E_grn (Guarani expert)",
}

# Default Jopara in-context prefix.
# Three short CEGPA-style CS phrases that cover all three regimes:
#   "ko'ápe" (Guaraní: "here"), "la gente" (Spanish), "o-vive" (jehe'a: live)
# The prefix primes the LLaMA decoder toward Jopara code-switching vocabulary.
DEFAULT_JOPARA_PREFIX = (
    "ko'ápe la gente o-vive, "
    "Paraguay-pe guaraní ha español o-ñe'ẽ, "
    "nde retã rehegua:"
)


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    n, m = len(r), len(h)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            curr[j] = prev[j-1] if r[i-1] == h[j-1] else 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    return prev[m] / n


def _embed_prefix(
    model,
    tok_encoder,
    prefix_text: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """
    Tokenize prefix_text and embed it through the LLaMA decoder embedding layer.

    Returns embedded tensor [B, L_prefix, D] or None if prefix is empty / fails.
    This is concatenated onto decoder_context before beam search so the first
    generated tokens are conditioned on the Jopara example text.
    """
    if not prefix_text.strip():
        return None
    try:
        token_ids = tok_encoder(prefix_text)          # [L_prefix] int64
        if token_ids.dim() == 0 or token_ids.numel() == 0:
            log.warning("Prefix tokenized to empty tensor; skipping prefix injection.")
            return None
        # Strip BOS/EOS if tokenizer added them — we only want content tokens
        bos_idx = getattr(model.target_vocab_info, "bos_idx", 0)
        eos_idx = getattr(model.target_vocab_info, "eos_idx", 2)
        mask = (token_ids != bos_idx) & (token_ids != eos_idx)
        token_ids = token_ids[mask]
        if token_ids.numel() == 0:
            return None

        token_ids = token_ids.to(device)              # [L_prefix]
        token_ids = token_ids.unsqueeze(0).expand(batch_size, -1)  # [B, L_prefix]

        # Embed through LLaMA decoder's input embedding (model_dim projection)
        emb_layer = model.llama_decoder.model.embed_tokens \
            if hasattr(model.llama_decoder, "model") \
            else model.llama_decoder.embed_tokens
        with torch.no_grad():
            embedded = emb_layer(token_ids).to(dtype)  # [B, L_prefix, D]
        log.info("Prefix '%s…' → %d tokens × dim %d", prefix_text[:40], token_ids.shape[1], embedded.shape[2])
        return embedded
    except Exception as exc:
        log.warning("Prefix embedding failed (%s); running without prefix.", exc)
        return None


def _transcribe_with_prefix(
    model,
    beam_search_generator,
    batch,
    prefix_embed: torch.Tensor | None,
    token_decoder,
) -> list[str]:
    """
    Replicate the non-streaming path of ASRInferencePipeline._apply_model_wav2vec2llama,
    injecting an optional embedded text prefix into the decoder context.
    """
    from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

    # Get audio-conditioned decoder context (same as standard pipeline)
    decoder_context, decoder_context_seq_lens, _ = model(batch, return_decoder_inputs=True)

    # Inject Jopara prefix by concatenating prefix embeddings after the audio context
    if prefix_embed is not None:
        B = decoder_context.shape[0]
        if prefix_embed.shape[0] != B:
            prefix_embed = prefix_embed[:B]  # truncate if batch ended early
        L_prefix = prefix_embed.shape[1]
        decoder_context = torch.cat([decoder_context, prefix_embed], dim=1)
        decoder_context_seq_lens = [l + L_prefix for l in decoder_context_seq_lens]

    hypothesis_tokens, hypothesis_lens = beam_search_generator.generate_hypotheses(
        decoder_context_inputs=[decoder_context],
        decoder_context_seq_lens=[decoder_context_seq_lens],
        audio_embeddings=None,
        batch=None,
    )

    transcriptions = []
    for i in range(hypothesis_tokens.shape[0]):
        seq_len = hypothesis_lens[i] if hypothesis_lens is not None else 0
        tokens = hypothesis_tokens[i, :seq_len]
        transcriptions.append(token_decoder(tokens))
    return transcriptions


def run_inference(
    model,
    beam_search_generator,
    pipeline,
    rows: list[dict],
    lang_mode: str,
    batch_size: int,
    gate_store: GateStore,
    tok_encoder,
    token_decoder,
    device: torch.device,
    dtype: torch.dtype,
    prefix_embed: torch.Tensor | None = None,
    include_expert_hyps: bool = True,
) -> list[dict]:
    """
    Run the patched J-MoLE model on each utterance. For each:
      1. Run full model (soft gates from router → gate_store), with optional prefix
      2. Optionally run each expert one-hot gate separately (no prefix, for fair comparison)
    """
    from fairseq2.datasets.batch import Seq2SeqBatch
    import torchaudio

    results: list[dict] = []
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        audio_paths = [r["audio_path"] for r in batch_rows]
        lang_tags   = [lang_mode] * len(batch_rows)

        # 1. Full J-MoLE inference with optional Jopara prefix
        gate_store.clear()
        try:
            if prefix_embed is not None:
                # Manual path: build batch ourselves then call _transcribe_with_prefix
                # Load + pad audio
                wavs = []
                for p in audio_paths:
                    wav, sr = torchaudio.load(p)
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    wavs.append(wav.squeeze(0))
                max_len = max(w.shape[0] for w in wavs)
                src = torch.zeros(len(wavs), max_len, dtype=torch.bfloat16)
                for j, w in enumerate(wavs):
                    # amplitude normalisation (same as pipeline)
                    w = w.float()
                    rms = w.pow(2).mean().sqrt().clamp(min=1e-8)
                    w = w / rms * 0.1
                    src[j, :w.shape[0]] = w.to(torch.bfloat16)
                src_lens = [w.shape[0] for w in wavs]

                # dummy target (BOS)
                tgt = torch.zeros(len(wavs), 1, dtype=torch.int64)
                batch_obj = Seq2SeqBatch(
                    source_seqs=src.to(device),
                    source_seq_lens=src_lens,
                    target_seqs=tgt.to(device),
                    target_seq_lens=[1] * len(wavs),
                    example={"lang": lang_tags},
                )
                B = len(batch_rows)
                pfx = prefix_embed[:B] if prefix_embed.shape[0] >= B else prefix_embed.expand(B, -1, -1)
                hyps_jmole = _transcribe_with_prefix(
                    model, beam_search_generator, batch_obj, pfx, token_decoder
                )
            else:
                hyps_jmole = pipeline.transcribe(audio_paths, lang=lang_tags)
                if isinstance(hyps_jmole, str):
                    hyps_jmole = [hyps_jmole]
        except Exception as e:
            log.warning("Batch %d-%d failed: %s", i, i + len(batch_rows), e)
            hyps_jmole = [""] * len(batch_rows)

        last_gates = gate_store.get()

        for j, (row, hyp_jmole) in enumerate(zip(batch_rows, hyps_jmole)):
            ref = row.get("reference", "")
            rec = {
                "utt_id":    row["utt_id"],
                "split":     row.get("split", "unknown"),
                "regime":    row.get("regime", ""),
                "reference": ref,
                "hyp_jmole": hyp_jmole,
                "wer_jmole": round(_wer(ref, hyp_jmole), 4) if ref else None,
                "expert_hyps": {},
            }
            if last_gates is not None:
                gates_j = last_gates if last_gates.dim() == 1 else (
                    last_gates[j] if j < last_gates.shape[0] else last_gates[-1]
                )
                rec["p_spa"] = round(gates_j[0].item(), 4)
                rec["p_cs"]  = round(gates_j[1].item(), 4)
                rec["p_grn"] = round(gates_j[2].item(), 4)
            else:
                rec["p_spa"] = rec["p_cs"] = rec["p_grn"] = None
            results.append(rec)

        # 2. One-hot expert inference (pseudo-N-best, no prefix for clean ablation)
        if include_expert_hyps:
            for idx, key in enumerate(EXPERT_KEYS):
                set_onehot_gate(gate_store, idx, next(model.parameters()).device)
                try:
                    hyps_k = pipeline.transcribe(audio_paths, lang=lang_tags)
                    if isinstance(hyps_k, str):
                        hyps_k = [hyps_k]
                except Exception as e:
                    log.warning("Expert %s batch %d failed: %s", key, i, e)
                    hyps_k = [""] * len(batch_rows)
                finally:
                    gate_store.unlock()

                for j, (row, hyp_k) in enumerate(zip(batch_rows, hyps_k)):
                    ref = row.get("reference", "")
                    results[-(len(batch_rows)) + j]["expert_hyps"][key] = {
                        "hyp": hyp_k,
                        "wer": round(_wer(ref, hyp_k), 4) if ref else None,
                    }

        done = min(i + batch_size, len(rows))
        if done % 50 == 0 or done == len(rows):
            elapsed = time.time() - t0
            wers = [r["wer_jmole"] for r in results if r["wer_jmole"] is not None]
            mean_wer = sum(wers) / len(wers) if wers else 0.0
            log.info("  %d/%d  (%.0fs, %.2fs/utt)  WER=%.1f%%",
                     done, len(rows), elapsed, elapsed / done, 100 * mean_wer)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/jmole_grn.yaml")
    parser.add_argument("--split",         choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--router-ckpt",   default=None,
                        help="A2-style router-only checkpoint (.pt). "
                             "Superseded by --a3-ckpt when both are given.")
    parser.add_argument("--a3-ckpt",       default=None,
                        help="A3 joint checkpoint (.pt). If given, loads router + "
                             "LoRA expert weights from this file. "
                             "Defaults to checkpoints/jmole_grn_a3/step_1200_best.pt "
                             "when --use-a3 is set.")
    parser.add_argument("--use-a3",        action="store_true",
                        help="Shorthand: load the A3 best checkpoint automatically.")
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--no-expert-hyps", action="store_true",
                        help="Skip per-expert one-hot inference (faster, no pseudo-N-best).")
    parser.add_argument("--jopara-prefix", action="store_true",
                        help="Inject a Jopara in-context prefix into the LLaMA decoder "
                             "before beam search. Biases toward CS vocabulary without "
                             "gradient updates. See DEFAULT_JOPARA_PREFIX in this file.")
    parser.add_argument("--prefix-text",   default=None,
                        help="Override the default Jopara prefix text (implies --jopara-prefix).")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parents[1] / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_mode = cfg.get("language_mode", "grn_Latn")

    # Resolve A3 checkpoint path
    a3_ckpt_path: Path | None = None
    if args.a3_ckpt:
        a3_ckpt_path = Path(args.a3_ckpt)
        if not a3_ckpt_path.is_absolute():
            a3_ckpt_path = J15 / args.a3_ckpt
    elif args.use_a3:
        a3_ckpt_path = J15 / "checkpoints/jmole_grn_a3/step_1200_best.pt"

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading J-MoLE model …")
    model, router, gate_store = build_jmole_model(cfg, stage="a3" if a3_ckpt_path else "a2")

    if a3_ckpt_path:
        if not a3_ckpt_path.exists():
            log.error("A3 checkpoint not found: %s", a3_ckpt_path)
            sys.exit(1)
        log.info("Loading A3 checkpoint: %s", a3_ckpt_path)
        ckpt = torch.load(a3_ckpt_path, map_location=device, weights_only=False)
        # Router — saved under key "router"
        router_sd = ckpt.get("router")
        if router_sd:
            router.load_state_dict(router_sd)
            log.info("  → Router weights loaded.")
        else:
            log.warning("  No 'router' key in checkpoint; router weights unchanged.")
        # LoRA experts — saved as {"mole_loras": {module_name: expert_state_dict}}
        mole_loras = ckpt.get("mole_loras", {})
        if mole_loras:
            n_loaded = 0
            for name, mod in model.named_modules():
                if isinstance(mod, MoELoRALinear) and name in mole_loras:
                    mod.load_expert_state_dict(mole_loras[name])
                    n_loaded += 1
            log.info("  → LoRA weights loaded into %d / %d MoELoRALinear layers.",
                     n_loaded, len(mole_loras))
        else:
            log.warning("  No 'mole_loras' key in checkpoint; LoRA weights unchanged.")
    elif args.router_ckpt:
        log.info("Loading router checkpoint: %s", args.router_ckpt)
        ckpt = torch.load(args.router_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt.get("router", ckpt))
        router.load_state_dict(sd)

    model.eval()
    router.eval()

    from fairseq2.data.tokenizers.hub import load_tokenizer
    from omnilingual_asr.models.inference import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaBeamSearchConfig
    from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
        Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
    )
    import omnilingual_asr.models.inference.pipeline as _asr_pipe
    _asr_pipe.MAX_ALLOWED_AUDIO_SEC = 120

    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    infer_dtype = _dtype_map.get(cfg.get("dtype", "float32"), torch.float32)

    tokenizer = load_tokenizer(cfg["model"]["card"])
    tok_encoder = tokenizer.create_encoder()
    tok_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    pipeline = ASRInferencePipeline(
        model_card=None, model=model, tokenizer=tokenizer,
        device=str(device), dtype=infer_dtype,
    )
    pipeline._model = model

    # Build a standalone beam search generator (mirrors pipeline internals)
    # so we can call it directly for prefix injection.
    beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
        model=model,
        config=Wav2Vec2LlamaBeamSearchConfig(nbest=1, length_norm=False),
        streaming_config=pipeline.streaming_config,
    )

    # ── Jopara prefix ─────────────────────────────────────────────────────────
    prefix_embed: torch.Tensor | None = None
    use_prefix = args.jopara_prefix or (args.prefix_text is not None)
    if use_prefix:
        prefix_text = args.prefix_text if args.prefix_text else DEFAULT_JOPARA_PREFIX
        log.info("Building Jopara prefix embedding: '%s'", prefix_text[:60])
        prefix_embed = _embed_prefix(
            model, tok_encoder, prefix_text,
            batch_size=args.batch_size,
            device=device, dtype=infer_dtype,
        )
        if prefix_embed is not None:
            log.info("Prefix embedding: shape %s", tuple(prefix_embed.shape))
        else:
            log.warning("Prefix embedding failed; running without prefix.")

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = MANIFEST_MAP.get(args.split)
    if manifest_path is None or not manifest_path.exists():
        log.error("Manifest not found for split=%s: %s", args.split, manifest_path)
        sys.exit(1)

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    log.info("Loaded %d utterances from %s", len(rows), manifest_path.name)

    # ── Run inference ─────────────────────────────────────────────────────────
    results = run_inference(
        model=model,
        beam_search_generator=beam_search_generator,
        pipeline=pipeline,
        rows=rows,
        lang_mode=lang_mode,
        batch_size=args.batch_size,
        gate_store=gate_store,
        tok_encoder=tok_encoder,
        token_decoder=tok_decoder,
        device=device,
        dtype=infer_dtype,
        prefix_embed=prefix_embed,
        include_expert_hyps=not args.no_expert_hyps,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = J15 / "jopara_mole_asr/hypothesis_packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = cfg.get("variant", "jmole_grn")
    stage_tag = "a3" if a3_ckpt_path else "a2"
    prefix_tag = "_prefix" if (use_prefix and prefix_embed is not None) else ""
    out_path = out_dir / f"{variant}_{stage_tag}_{args.split}_hypotheses{prefix_tag}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(results), out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    wers = [r["wer_jmole"] for r in results if r["wer_jmole"] is not None]
    if wers:
        print(f"\n=== J-MoLE-ASR Inference ({stage_tag.upper()}, {args.split}"
              f"{', +prefix' if prefix_tag else ''}) ===")
        print(f"Utterances:    {len(results)}")
        print(f"J-MoLE WER:    {100*sum(wers)/len(wers):.2f}%")
        if not args.no_expert_hyps:
            for key in EXPERT_KEYS:
                expert_wers = [
                    r["expert_hyps"][key]["wer"]
                    for r in results
                    if key in r["expert_hyps"] and r["expert_hyps"][key]["wer"] is not None
                ]
                if expert_wers:
                    print(f"E_{key} WER:     {100*sum(expert_wers)/len(expert_wers):.2f}%")
        print(f"Output:        {out_path}")


if __name__ == "__main__":
    main()
