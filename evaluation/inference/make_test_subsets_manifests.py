#!/usr/bin/env python3
"""
Build eval manifests for the 4 test subsets (test_guarani_only, test_spanish_only, test_mixed, test_jehea).
Gold label priority: final_corrected_reference_norm → corrected_reference_norm → reference_norm.
Compatible with both old test_subsets/ and v3 v2_test_subsets/ (which have final_corrected_reference_norm).
Audio path: If --audio-root is set, we resolve using the CSV row's full basename (e.g. tr-008_00426.wav) so that
each (session, segment) pair maps to the correct file. We look for <audio-root>/<basename> first; if not found,
fall back to segment-only match *<segment>.wav (for flat dirs with a single session). If --audio-root is unset,
use the CSV directory column as-is.
Writes .tsv and .wrd per subset for use with eval_debug_manifest.py --split.
Language column: empty by default so the OmniASR pipeline auto-detects language per utterance (the paper's
row-3 setting). Earlier versions hardcoded spa_Latn here, which silently forced Spanish decoding.
--combined-split NAME additionally writes NAME.tsv/.wrd/.ids covering all subsets (ids = "<utt_id>\t<v2_category>").
The Table 1 test manifest (results/rebuttal_stats/row3_baseline/manifests/v2_test_all.*) is reproduced by:
  --test-subsets-dir updated_csvs/v2_test_subsets --combined-split v2_test_all --max-duration 0
"""
import argparse
import csv
import re
from pathlib import Path


def one_line(s: str) -> str:
    if not s:
        return s
    return re.sub(r"[\r\n]+", " ", s).strip()


def main():
    p = argparse.ArgumentParser(description="Build test subset manifests from test_subsets/*.csv")
    p.add_argument("--test-subsets-dir", type=Path, required=True,
                   help="Directory containing test_guarani_only.csv, test_spanish_only.csv, test_mixed.csv, test_jehea.csv")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for manifests (test_*.tsv, test_*.wrd)")
    p.add_argument("--audio-root", type=Path, default=None,
                   help="Directory containing WAVs; resolve by full basename (e.g. tr-008_00426.wav) so CSV session id is preserved. If unset, use CSV directory as-is.")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip rows whose audio file does not exist (e.g. when paths are visible at build time)")
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing", default=False,
                   help="Include all CSV rows even if audio path does not exist (default; for paths on Slate when building from project)")
    p.add_argument("--max-duration", type=float, default=40.0,
                   help="Skip rows whose duration (seconds) exceeds this value (default: 40.0 — OmniASR pipeline hard cap; 0 disables)")
    p.add_argument("--lang", type=str, default="",
                   help="Language code for the manifest's 3rd column (default: empty => pipeline auto-detects language)")
    p.add_argument("--combined-split", type=str, default="",
                   help="If set, also write <name>.tsv/.wrd/.ids with all subsets, in order guarani_only, jehea, mixed, spanish_only")
    args = p.parse_args()

    subset_files = [
        ("test_guarani_only", "test_guarani_only.csv"),
        ("test_spanish_only", "test_spanish_only.csv"),
        ("test_mixed", "test_mixed.csv"),
        ("test_jehea", "test_jehea.csv"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {}  # split_name -> [(path, ref, utt_id, category)]

    for split_name, csv_name in subset_files:
        csv_path = args.test_subsets_dir / csv_name
        if not csv_path.exists():
            print("Skipping {} (not found)".format(csv_path))
            continue
        rows = []
        skipped_duration = 0
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                directory = (row.get("directory") or "").strip()
                # Gold label: final_corrected_reference_norm (v3) → corrected_reference_norm → reference_norm
                ref = one_line((row.get("final_corrected_reference_norm") or row.get("corrected_reference_norm") or row.get("reference_norm") or "").strip())
                if not directory or not ref:
                    continue
                # Skip utterances that exceed the OmniASR pipeline's max audio length
                try:
                    dur = float(row.get("duration") or 0)
                    if args.max_duration > 0 and dur > args.max_duration:
                        print("  SKIP (duration {:.1f}s > {:.0f}s cap): {}".format(dur, args.max_duration, Path(directory).name))
                        skipped_duration += 1
                        continue
                except (ValueError, TypeError):
                    pass
                # Preserve full basename (e.g. tr-008_00426.wav) so (session, segment) mapping matches the CSV.
                basename = Path(directory).name
                segment_wav = basename.split("_")[-1] if "_" in basename else basename  # e.g. 02472.wav
                if args.audio_root is not None:
                    # 1) Resolve by full basename so tr-008_00426.wav in CSV -> same file under audio-root
                    path_by_basename = (args.audio_root / basename).resolve()
                    if path_by_basename.exists():
                        path = path_by_basename
                    else:
                        # 2) Fallback: segment id only (for flat dirs with a single session)
                        direct = (args.audio_root / segment_wav).resolve()
                        if direct.exists():
                            path = direct
                        else:
                            matches = list(args.audio_root.glob("*" + segment_wav))
                            path = Path(matches[0]).resolve() if len(matches) >= 1 else path_by_basename
                else:
                    path = Path(directory).resolve()
                if args.skip_missing and not path.exists():
                    continue
                category = (row.get("v2_category") or split_name[len("test_"):]).strip()
                rows.append((str(path), ref, Path(basename).stem, category))
        if not rows:
            print("No rows for {}".format(split_name))
            continue
        write_manifest(args.output_dir, split_name, rows, args.lang)
        combined[split_name] = rows
        msg = "Wrote {} ({} utterances) -> {}".format(split_name, len(rows), args.output_dir)
        if skipped_duration:
            msg += "  [{} skipped: duration > {:.0f}s]".format(skipped_duration, args.max_duration)
        print(msg)

    if args.combined_split:
        order = ["test_guarani_only", "test_jehea", "test_mixed", "test_spanish_only"]
        rows = [r for name in order for r in combined.get(name, [])]
        write_manifest(args.output_dir, args.combined_split, rows, args.lang, with_ids=True)
        print("Wrote {} ({} utterances) -> {}".format(args.combined_split, len(rows), args.output_dir))


def write_manifest(output_dir, split_name, rows, lang, with_ids=False):
    tsv = output_dir / (split_name + ".tsv")
    wrd = output_dir / (split_name + ".wrd")
    with tsv.open("w", encoding="utf-8") as ft, wrd.open("w", encoding="utf-8") as fw:
        ft.write("/\n")
        for path_str, ref, _, _ in rows:
            ft.write("{}\t0\t{}\n".format(path_str, lang))
            fw.write(ref + "\n")
    if with_ids:
        with (output_dir / (split_name + ".ids")).open("w", encoding="utf-8") as fi:
            for _, _, utt_id, category in rows:
                fi.write("{}\t{}\n".format(utt_id, category))


if __name__ == "__main__":
    main()
