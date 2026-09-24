#!/usr/bin/env python3
"""Evaluate dev split with CER/WER using omnilingual-asr inference (default 7B model)."""
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

# LoRA checkpoints were trained with lora_7b_spa (inject .linear, .lora_A, .lora_B into decoder)
# .../Jopara_ASR_models/results/Guarani_only_result/finetune_llm_7b/scripts/eval_debug_manifest.py -> Jopara_ASR_models
LORA_7B_SPA_ROOT = Path(__file__).resolve().parents[2] / "training" / "lora_adapters"
if LORA_7B_SPA_ROOT.exists():
    sys.path.insert(0, str(LORA_7B_SPA_ROOT))
try:
    from lora_inject import inject_lora
except ImportError:
    inject_lora = None


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def word_error_stats(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return (1.0, len(hyp_words), 0, 0, 0, len(hyp_words)) if hyp_words else (0.0, 0, 0, 0, 0, 0)
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        dp[0][j] = (j, j, 0, 0)
    for i in range(1, rows):
        for j in range(1, cols):
            ref_word = ref_words[i - 1]
            hyp_word = hyp_words[j - 1]
            if ref_word == hyp_word:
                cand = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                cand = (sub[0] + 1, sub[1], sub[2], sub[3] + 1)
            ins = dp[i][j - 1]
            ins_cand = (ins[0] + 1, ins[1] + 1, ins[2], ins[3])
            delete = dp[i - 1][j]
            del_cand = (delete[0] + 1, delete[1], delete[2] + 1, delete[3])
            dp[i][j] = min(cand, ins_cand, del_cand, key=lambda x: (x[0], x[1], x[2], x[3]))
    dist, ins, delete, sub = dp[-1][-1]
    return dist / float(len(ref_words)), ins, delete, sub, len(ref_words), len(hyp_words)


def read_manifest(manifest_dir, split):
    tsv = manifest_dir / (split + ".tsv")
    wrd = manifest_dir / (split + ".wrd")
    if not tsv.exists() or not wrd.exists():
        raise FileNotFoundError("Missing {} manifest in {}".format(split, manifest_dir))
    with tsv.open(encoding="utf-8") as f:
        rows = [line.rstrip().split("\t") for line in f.readlines()[1:] if line.strip()]
    audio_paths = [r[0] for r in rows]
    langs = [r[2] if len(r) > 2 else "" for r in rows]
    with wrd.open(encoding="utf-8") as f:
        texts = [line.rstrip() for line in f if line.strip()]
    if len(audio_paths) != len(texts):
        raise RuntimeError("TSV/WRD size mismatch")
    return audio_paths, texts, langs


def main():
    p = argparse.ArgumentParser(description="Evaluate dev split with CER/WER.")
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--split", type=str, default="dev", help="Manifest split name (e.g. dev, test_guarani_only, test_spanish_only)")
    p.add_argument("--model-card", type=str, default="omniASR_LLM_7B_v2")
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to trained checkpoint. If set, load this instead of the model card default (required after fine-tuning).")
    p.add_argument("--fallback-lang", type=str, default="")
    p.add_argument("--override-lang", type=str, default="", help="If set, use this language for all utterances (overrides manifest and fallback-lang).")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    audio_paths, references, langs = read_manifest(args.manifest_dir, args.split)

    # Filter to existing, non-empty audio so the pipeline does not get zero-length data
    valid = []
    for path, ref, lang in zip(audio_paths, references, langs):
        p = Path(path)
        if not p.exists():
            continue
        try:
            if p.stat().st_size == 0:
                continue
        except OSError:
            continue
        valid.append((path, ref, lang))
    if len(valid) < len(audio_paths):
        n_skip = len(audio_paths) - len(valid)
        print("Warning: skipping {} path(s) (missing or empty). Run where AUDIO_ROOT is visible (e.g. Slate).".format(n_skip), file=sys.stderr)
    if not valid:
        raise FileNotFoundError(
            "No valid audio files found. Manifest paths may be on Slate (e.g. /N/slate/...). "
            "Set AUDIO_ROOT and rebuild manifests where audio is visible, or run this script on a node where those paths exist."
        )
    audio_paths, references, langs = [t[0] for t in valid], [t[1] for t in valid], [t[2] for t in valid]

    if args.checkpoint is not None and args.checkpoint.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hub = get_wav2vec2_llama_model_hub()
        # LoRA checkpoints from lora_7b_spa save q_proj.linear, lora_A, lora_B (injected into decoder).
        # Load base model, inject LoRA from run's lora_config.json, then load checkpoint state dict.
        # Use float16 for LoRA inference: fairseq2 position_encoder uses view_as_complex which does not support bfloat16.
        if inject_lora is not None:
            dtype = torch.float16
            print("[eval] Loading base model (this may take 1–2 min)...", flush=True, file=sys.stderr)
            model = hub.load_model(args.model_card, device=device, dtype=dtype)
            ckpt = args.checkpoint.resolve()
            run_dir = ckpt.parent.parent.parent.parent.parent.parent.parent  # .../tp_00 -> A0_first_batch
            lora_config_path = run_dir / "lora_config.json"
            print("[eval] Injecting LoRA and loading checkpoint...", flush=True, file=sys.stderr)
            if lora_config_path.exists():
                with lora_config_path.open(encoding="utf-8") as f:
                    lora_cfg = json.load(f)
                r = int(lora_cfg.get("r", 16))
                alpha = float(lora_cfg.get("lora_alpha", 32))
                dropout = float(lora_cfg.get("lora_dropout", 0.05))
                target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
                inject_lora(model, target_modules=target_modules, r=r, alpha=alpha, dropout=dropout)
            else:
                inject_lora(model, target_modules=["q_proj", "v_proj"], r=16, alpha=32.0, dropout=0.05)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
            # LoRA params load as float32; cast full model to inference dtype so q_proj(seqs) and lora_A/lora_B match
            model = model.to(device=device, dtype=dtype)
            print("[eval] Checkpoint loaded. Building pipeline...", flush=True, file=sys.stderr)
        else:
            dtype = torch.bfloat16
            print("[eval] Loading custom checkpoint...", flush=True, file=sys.stderr)
            config = hub.get_model_config(args.model_card)
            model = hub.load_custom_model(args.checkpoint, config, device=device, dtype=dtype)
        tokenizer = load_tokenizer(args.model_card)
        pipeline = ASRInferencePipeline(model_card=None, model=model, tokenizer=tokenizer, device=device, dtype=dtype)
    else:
        pipeline = ASRInferencePipeline(model_card=args.model_card)
    if args.override_lang:
        lang_list = [args.override_lang] * len(audio_paths)
    else:
        lang_list = [lang or args.fallback_lang for lang in langs]
    n_utts = len(audio_paths)
    print("[eval] Transcribing {} utterances (batch_size={}, may take several min)...".format(n_utts, args.batch_size), flush=True, file=sys.stderr)
    if any(lang_list):
        hyps = pipeline.transcribe(audio_paths, lang=lang_list, batch_size=args.batch_size)
    else:
        hyps = pipeline.transcribe(audio_paths, batch_size=args.batch_size)
    print("[eval] Done transcribing. Writing results...", flush=True, file=sys.stderr)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "reference", "hypothesis", "cer", "wer", "ins", "del", "sub", "ref_words", "hyp_words"]
    total_cer = total_wer = total_ins = total_del = total_sub = 0.0
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for path, ref, hyp in zip(audio_paths, references, hyps):
            ref_n = normalize_text(ref)
            hyp_n = normalize_text(hyp)
            cer = levenshtein(ref_n, hyp_n) / float(len(ref_n)) if ref_n else 0.0
            wer, ins, delete, sub, rw, hw = word_error_stats(ref_n, hyp_n)
            total_cer += cer
            total_wer += wer
            total_ins += ins
            total_del += delete
            total_sub += sub
            w.writerow({
                "audio_path": path, "reference": ref, "hypothesis": hyp,
                "cer": round(cer, 4), "wer": round(wer, 4),
                "ins": ins, "del": delete, "sub": sub, "ref_words": rw, "hyp_words": hw,
            })
    if audio_paths:
        n = len(audio_paths)
        print("Avg CER={:.4f}, Avg WER={:.4f}, ins/del/sub={}/{}/{}".format(
            total_cer / n, total_wer / n, total_ins, total_del, total_sub))


if __name__ == "__main__":
    main()
