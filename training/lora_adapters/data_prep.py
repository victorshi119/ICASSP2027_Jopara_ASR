#!/usr/bin/env python3
"""
Prepare training manifest with lang=spa_Latn for all utterances (unified Jopara hypothesis).
Reads by_quality/train.csv, resolves audio paths under --audio-root, writes manifest TSV+WRD
suitable for LoRA training. Language conditioning: always spa_Latn.
"""
import argparse
import csv
import os
from pathlib import Path


def find_audio_by_basename(audio_root: Path, basename: str) -> str:
    direct = audio_root / basename
    if direct.is_file():
        return str(direct)
    for parent, _dirs, files in os.walk(audio_root):
        if basename in files:
            return str(Path(parent) / basename)
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=None,
                   help="Default: by_quality/train.csv next to this script's parent")
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lang", type=str, default="spa_Latn", help="Force this lang for all (default spa_Latn)")
    args = p.parse_args()
    if args.train_csv is None:
        script_dir = Path(__file__).resolve().parent
        by_quality = script_dir.parent / "preprocessed_data" / "audio_segments_full_v3_16k" / "updated_csvs" / "by_quality"
        if not by_quality.exists():
            by_quality = script_dir.parent / "by_quality"
        args.train_csv = by_quality / "train.csv"
    if not args.train_csv.exists():
        raise SystemExit(f"Train CSV not found: {args.train_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.train_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            directory = (row.get("directory") or "").strip()
            text = (row.get("corrected_reference_norm") or row.get("reference_norm") or "").strip()
            if not directory or not text:
                continue
            path = find_audio_by_basename(args.audio_root, Path(directory).name)
            if not path:
                continue
            rows.append({"path": path, "text": text, "lang": args.lang})
    tsv_path = args.output_dir / "train.tsv"
    wrd_path = args.output_dir / "train.wrd"
    with tsv_path.open("w", encoding="utf-8") as ft, wrd_path.open("w", encoding="utf-8") as fw:
        ft.write("/\n")
        for r in rows:
            ft.write(f"{r['path']}\t0\t{r['lang']}\n")
            fw.write(r["text"].replace("\n", " ").strip() + "\n")
    print(f"Wrote {len(rows)} rows to {tsv_path} and {wrd_path} (lang={args.lang} for all)")


if __name__ == "__main__":
    main()
