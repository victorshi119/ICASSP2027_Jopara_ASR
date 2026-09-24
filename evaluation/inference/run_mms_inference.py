#!/usr/bin/env python3
"""
MMS-1B-all baseline inference on CEGPA test subsets (ICASSP 2027 essay,
2026-09-12): the tokenizer-based-vs-non-tokenizer comparison needs a second
tokenizer-based baseline alongside Whisper. MMS (facebook/mms-1b-all) is
wav2vec2 + CTC with a character-level output head and a per-language adapter
(58 symbols for the Guarani adapter, confirmed by loading it) -- so, like
Whisper, it needs an explicit target language per run; unlike Whisper there
is no "auto" mode, so this runs both the Spanish and Guarani adapters across
all four subsets, mirroring Whisper's auto/spa split as closely as MMS's API
allows.

Reads the same TSV/WRD manifests used by the Whisper and OmniASR baselines,
writes eval CSVs with (audio_path, reference, hypothesis) columns, then calls
metrics.py via subprocess (sclite) -- identical eval pipeline to both.

Usage:
  python run_mms_inference.py [--mode grn|spa|all] [--subsets ...]
"""
from __future__ import annotations

import os
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
MANIFEST_DIR = PROJECT_ROOT / "results/June_09_baseline/manifests"
MMS_OUT      = PROJECT_ROOT / "results/June_09_baseline/mms"
METRICS_PY   = PROJECT_ROOT / "metrics.py"

SUBSETS = ["test_guarani_only", "test_spanish_only", "test_mixed", "test_jehea"]

# MMS has no auto-detect mode (unlike Whisper): each run picks one language
# adapter and applies it to every utterance, same convention as the paper's
# grn_Latn / spa_Latn zero-shot mode rows for OmniASR.
LANG_MODES: dict[str, str] = {
    "spa": "spa",
    "grn": "grn",
}

MODEL_ID = "facebook/mms-1b-all"


# ---------------------------------------------------------------------------
# Manifest reading (identical to run_whisper_inference.py)
# ---------------------------------------------------------------------------

def read_manifest(subset: str) -> tuple[list[str], list[str]]:
    tsv_path = MANIFEST_DIR / f"{subset}.tsv"
    wrd_path = MANIFEST_DIR / f"{subset}.wrd"

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    audio_paths = [line.split("\t")[0] for line in lines[1:] if line.strip()]
    references = wrd_path.read_text(encoding="utf-8").splitlines()

    if len(audio_paths) != len(references):
        sys.exit(
            f"ERROR {subset}: {len(audio_paths)} audio paths vs "
            f"{len(references)} references"
        )
    return audio_paths, references


# ---------------------------------------------------------------------------
# Inference -> eval CSV
# ---------------------------------------------------------------------------

def transcribe_one(model, processor, audio_path: str, device: str) -> str:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16kHz audio, got {sr}Hz for {audio_path}")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids).strip()


def run_inference(
    model, processor, audio_paths: list[str], references: list[str],
    lang_label: str, device: str, eval_csv_path: Path,
) -> None:
    rows = []
    for i, (audio_path, ref) in enumerate(zip(audio_paths, references), 1):
        print(f"  [{i}/{len(audio_paths)}] {Path(audio_path).name}  lang={lang_label}",
              flush=True)
        try:
            hyp = transcribe_one(model, processor, audio_path, device)
        except Exception as e:
            print(f"    WARNING: transcription failed: {e}", file=sys.stderr)
            hyp = ""
        rows.append({"audio_path": audio_path, "reference": ref, "hypothesis": hyp})

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference", "hypothesis"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {eval_csv_path}")


# ---------------------------------------------------------------------------
# Predictions text files + sclite metrics (identical to run_whisper_inference.py)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip())


def build_predictions(eval_csv_path: Path, subset: str, pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    refs, hyps = [], []
    with eval_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = (row.get("reference") or "").strip()
            hyp = (row.get("hypothesis") or "").strip()
            refs.append(normalize(ref))
            hyps.append(normalize(hyp))

    ref_path = pred_dir / f"{subset}.ref.txt"
    hyp_path = pred_dir / f"{subset}.hyp.txt"
    ref_path.write_text("\n".join(refs), encoding="utf-8")
    hyp_path.write_text("\n".join(hyps), encoding="utf-8")
    print(f"  Predictions: {ref_path.name} + {hyp_path.name}  ({len(refs)} utts)")


def compute_metrics(subset: str, pred_dir: Path, metrics_json: Path) -> None:
    ref_path = pred_dir / f"{subset}.ref.txt"
    hyp_path = pred_dir / f"{subset}.hyp.txt"
    cmd = [
        sys.executable, str(METRICS_PY),
        "--ref", str(ref_path),
        "--hyp", str(hyp_path),
        "--subset", subset,
        "--output", str(metrics_json),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: metrics.py exited {result.returncode} for {subset}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(LANG_MODES.keys()) + ["all"], default="all")
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    args = ap.parse_args()

    modes = list(LANG_MODES.keys()) if args.mode == "all" else [args.mode]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
    print("Model loaded.\n", flush=True)

    for mode_name in modes:
        lang = LANG_MODES[mode_name]
        print(f"Loading adapter for '{lang}' ...", flush=True)
        processor.tokenizer.set_target_lang(lang)
        model.load_adapter(lang)

        mode_dir = MMS_OUT / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = mode_dir / "predictions"
        metrics_json = mode_dir / "metrics.json"

        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}  (MMS adapter={lang})")
        print(f"{'='*60}\n")

        for subset in args.subsets:
            print(f"--- {subset} ---")
            audio_paths, references = read_manifest(subset)
            print(f"  {len(audio_paths)} utterances")

            eval_csv = mode_dir / f"{subset}_eval.csv"
            run_inference(model, processor, audio_paths, references, lang, device, eval_csv)
            build_predictions(eval_csv, subset, pred_dir)
            compute_metrics(subset, pred_dir, metrics_json)
            print()

    print("All done.")


if __name__ == "__main__":
    main()
