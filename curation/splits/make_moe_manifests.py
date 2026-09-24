#!/usr/bin/env python3
"""
Build MoE JSONL manifests from CEGPA test/dev CSVs.

Reads June2026_test_subsets (or other v2-style CSVs) and produces
unified JSONL manifests with JS2-compatible audio paths and dominance labels.

Usage (run from Jopara_ASR_models/):
    source env.sh
    $OMNI_PY moe_lora/scripts/make_moe_manifests.py \
        --test-root updated_csvs/v2_test_subsets \
        --dev-root  updated_csvs/training_splits_v3 \
        --audio-root $BASE/audio_segments_full_16k \
        --out-test  moe_lora/manifests/v2_test_moe_manifest.jsonl \
        --out-dev   moe_lora/manifests/v3_dev_moe_manifest.jsonl

Note: dev manifest will have 217 rows until CV Guaraní MP3s (gn/clips/) are resampled
to WAV at gn/clips_wav_16k/. Test manifest (150 rows) is always complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def derive_dominance(n_s: int, n_g: int, n_j: int) -> str:
    spa = float(n_s) + 0.5 * float(n_j)
    grn = float(n_g) + 0.5 * float(n_j)
    total = spa + grn
    if total <= 0:
        return "unknown"
    if spa / total >= 0.65:
        return "spa_dominant"
    if grn / total >= 0.65:
        return "grn_dominant"
    return "balanced"


def derive_gold_expert(category: str, dominance: str) -> str:
    if category == "spanish_only":
        return "E_spa_only"
    if category == "guarani_only":
        return "E_grn_only"
    if category in ("mixed", "jehe'a"):
        if dominance == "spa_dominant":
            return "E_spa_mixed"
        return "E_grn_mixed"
    return "E_grn_mixed"


def resolve_js2_audio(row: dict, audio_root: Path) -> str:
    """Find audio file by basename from ev_directory or directory column."""
    for col in ("ev_directory", "directory"):
        val = (row.get(col) or "").strip()
        if val:
            basename = Path(val).name
            candidate = audio_root / basename
            if candidate.exists():
                return str(candidate)
    return ""


def build_manifest(csv_paths: list[Path], audio_root: Path, split: str) -> list[dict]:
    rows_out = []
    missing_audio = []

    for csv_path in csv_paths:
        subset_name = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = (row.get("final_corrected_reference_norm") or "").strip()
                if not ref:
                    continue

                quality = (row.get("quality_comments") or "").strip()

                audio_path = resolve_js2_audio(row, audio_root)
                if not audio_path:
                    ev = row.get("ev_directory", row.get("directory", ""))
                    missing_audio.append(Path(ev).name if ev else "?")
                    continue

                ev = row.get("ev_directory", row.get("directory", ""))
                utt_id = Path(ev).stem if ev else Path(audio_path).stem

                n_s = int(row.get("spanish_word_count") or row.get("v2_spa_count") or 0)
                n_g = int(row.get("guarani_word_count") or row.get("v2_grn_count") or 0)
                n_j = int(row.get("jehe_a_word_count") or row.get("v2_jha_count") or 0)
                n_o = int(row.get("total_word_count") or row.get("v2_other_count") or 0)

                category = (row.get("v2_category") or row.get("new_category") or "unknown").strip()
                dominance = derive_dominance(n_s, n_g, n_j)
                gold_expert = derive_gold_expert(category, dominance)

                try:
                    duration = float(row.get("duration") or 0)
                except ValueError:
                    duration = 0.0

                session = (row.get("file_id") or "").strip()

                rows_out.append({
                    "utt_id": utt_id,
                    "subset": subset_name,
                    "split": split,
                    "audio_path": audio_path,
                    "duration_sec": duration,
                    "reference": ref,
                    "reference_norm": ref,
                    "v2_category": category,
                    "v2_spa_count": n_s,
                    "v2_grn_count": n_g,
                    "v2_jha_count": n_j,
                    "v2_other_count": n_o,
                    "dominance_gold": dominance,
                    "router_gold_expert": gold_expert,
                    "quality_comments": quality,
                    "session": session,
                    "source_csv": str(csv_path),
                })

    if missing_audio:
        print(f"  WARNING: {len(missing_audio)} rows had no matching audio in {audio_root}:")
        for m in missing_audio[:10]:
            print(f"    {m}")
        if len(missing_audio) > 10:
            print(f"    ... and {len(missing_audio) - 10} more")

    return rows_out


def main():
    p = argparse.ArgumentParser(description="Build MoE JSONL manifests from CEGPA CSVs.")
    p.add_argument("--test-root", type=Path,
                   default=Path("updated_csvs/v2_test_subsets"),
                   help="Directory containing test_*.csv files")
    p.add_argument("--dev-root", type=Path,
                   default=Path("updated_csvs/training_splits_v3"),
                   help="Directory containing dev_*.csv files")
    p.add_argument("--audio-root", type=Path,
                   default=Path("/media/volume/sapc2_data/victor/icassp2026/Jopara_ASR_models/audio_segments_full_16k"))
    p.add_argument("--out-test", type=Path,
                   default=Path("moe_lora/manifests/v2_test_moe_manifest.jsonl"))
    p.add_argument("--out-dev", type=Path,
                   default=Path("moe_lora/manifests/v3_dev_moe_manifest.jsonl"))
    p.add_argument("--skip-dev", action="store_true",
                   help="Only build the test manifest (skip dev)")
    args = p.parse_args()

    if not args.audio_root.exists():
        raise SystemExit(f"Audio root not found: {args.audio_root}")

    # ── Test manifest ────────────────────────────────────────────────────────
    test_csvs = sorted(args.test_root.glob("test_*.csv")) if args.test_root.exists() else []
    if not test_csvs:
        print(f"WARNING: no test_*.csv files found under {args.test_root}")
    else:
        print(f"Building test manifest from {len(test_csvs)} CSVs in {args.test_root} ...")
        test_rows = build_manifest(test_csvs, args.audio_root, split="test")
        args.out_test.parent.mkdir(parents=True, exist_ok=True)
        with args.out_test.open("w", encoding="utf-8") as f:
            for r in test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {len(test_rows)} rows written to {args.out_test}")

        # sanity check
        subsets = {}
        for r in test_rows:
            subsets[r["subset"]] = subsets.get(r["subset"], 0) + 1
        for s, n in sorted(subsets.items()):
            print(f"     {s}: {n}")

    if args.skip_dev:
        return

    # ── Dev manifest ─────────────────────────────────────────────────────────
    dev_csvs = []
    if args.dev_root.exists():
        dev_csvs = sorted(args.dev_root.glob("dev_*.csv"))
    if not dev_csvs:
        print(f"WARNING: no dev_*.csv files found under {args.dev_root} — skipping dev manifest")
        return

    # Dev CSVs may have different column schema (no ev_directory, no v2_category).
    # Only include rows with resolvable audio AND non-empty reference.
    print(f"Building dev manifest from {len(dev_csvs)} CSVs in {args.dev_root} ...")
    dev_rows = build_manifest(dev_csvs, args.audio_root, split="dev")
    args.out_dev.parent.mkdir(parents=True, exist_ok=True)
    with args.out_dev.open("w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {len(dev_rows)} rows written to {args.out_dev}")


if __name__ == "__main__":
    main()
