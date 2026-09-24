#!/usr/bin/env python3
"""
Whisper large-v3 baseline inference on CEGPA test subsets.

Reads the same TSV/WRD manifests used by the OmniASR baseline, writes eval
CSVs with (audio_path, reference, hypothesis) columns, then calls metrics.py
via subprocess (sclite) — identical eval pipeline to OmniASR.

Language modes:
  auto  → language=None  (Whisper auto-detects per utterance)
  spa   → language="es"  (force Spanish)
"""
from __future__ import annotations

import os
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import whisper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
MANIFEST_DIR = PROJECT_ROOT / "results/June_09_baseline/manifests"
WHISPER_OUT  = PROJECT_ROOT / "results/June_09_baseline/whisper_large"
METRICS_PY   = PROJECT_ROOT / "metrics.py"

SUBSETS = ["test_guarani_only", "test_spanish_only", "test_mixed", "test_jehea"]

LANG_MODES: dict[str, str | None] = {
    "auto": None,   # Whisper auto language detection
    "spa":  "es",   # Force Spanish
}


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def read_manifest(subset: str) -> tuple[list[str], list[str]]:
    """Return (audio_paths, references) from the TSV+WRD manifest pair."""
    tsv_path = MANIFEST_DIR / f"{subset}.tsv"
    wrd_path  = MANIFEST_DIR / f"{subset}.wrd"

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    # First line is the root directory header — skip it
    audio_paths = [line.split("\t")[0] for line in lines[1:] if line.strip()]

    references = wrd_path.read_text(encoding="utf-8").splitlines()

    if len(audio_paths) != len(references):
        sys.exit(
            f"ERROR {subset}: {len(audio_paths)} audio paths vs "
            f"{len(references)} references"
        )
    return audio_paths, references


# ---------------------------------------------------------------------------
# Inference → eval CSV
# ---------------------------------------------------------------------------

def run_inference(
    model: whisper.Whisper,
    audio_paths: list[str],
    references: list[str],
    language: str | None,
    eval_csv_path: Path,
) -> None:
    rows = []
    lang_label = language if language else "auto-detect"
    for i, (audio_path, ref) in enumerate(zip(audio_paths, references), 1):
        print(f"  [{i}/{len(audio_paths)}] {Path(audio_path).name}  lang={lang_label}",
              flush=True)
        try:
            result = model.transcribe(audio_path, language=language)
            hyp = result["text"].strip()
        except Exception as e:
            print(f"    WARNING: transcription failed: {e}", file=sys.stderr)
            hyp = ""
        rows.append({"audio_path": audio_path, "reference": ref, "hypothesis": hyp})

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "reference", "hypothesis"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {eval_csv_path}")


# ---------------------------------------------------------------------------
# Predictions text files + sclite metrics (mirrors OmniASR sbatch pipeline)
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
            ref = (
                row.get("corrected_reference_norm")
                or row.get("reference_norm")
                or row.get("reference")
                or ""
            ).strip()
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
        "--ref",    str(ref_path),
        "--hyp",    str(hyp_path),
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
    ap.add_argument("--model-size", default="large-v3",
                    help="Whisper model size (default: large-v3)")
    ap.add_argument(
        "--mode",
        choices=list(LANG_MODES.keys()) + ["all"],
        default="all",
        help="Language mode: auto, spa, or all (default: all)",
    )
    ap.add_argument("--subsets", nargs="+", default=SUBSETS,
                    help="Subsets to evaluate (default: all four)")
    args = ap.parse_args()

    modes = list(LANG_MODES.keys()) if args.mode == "all" else [args.mode]

    print(f"Loading whisper/{args.model_size} ...", flush=True)
    model = whisper.load_model(args.model_size)
    print("Model loaded.\n", flush=True)

    for mode_name in modes:
        language = LANG_MODES[mode_name]
        mode_dir = WHISPER_OUT / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        pred_dir    = mode_dir / "predictions"
        metrics_json = mode_dir / "metrics.json"

        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}  (language={'auto-detect' if language is None else language})")
        print(f"{'='*60}\n")

        for subset in args.subsets:
            print(f"--- {subset} ---")
            audio_paths, references = read_manifest(subset)
            print(f"  {len(audio_paths)} utterances")

            eval_csv = mode_dir / f"{subset}_eval.csv"
            run_inference(model, audio_paths, references, language, eval_csv)
            build_predictions(eval_csv, subset, pred_dir)
            compute_metrics(subset, pred_dir, metrics_json)
            print()

    print("All done.")


if __name__ == "__main__":
    main()
