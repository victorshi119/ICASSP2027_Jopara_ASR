#!/usr/bin/env python3
"""
Build manifests from train_90.csv and dev_10.csv.
Writes train, dev (and train_all, train_spanish, train_guarani, train_mixed) to output-dir/manifests
(or output-dir/<manifest-subdir> when --manifest-subdir is set).
Column 3 = lang (BCP 47, e.g. spa_Latn, grn_Latn, gug_Latn). Column 4 = expert label for routing.
Use --override-lang to force one language code for all rows (e.g. gug_Latn for full pipeline in that mode).
"""
import argparse
import csv
import re
import wave
from pathlib import Path


def one_line(s):
    if not s:
        return s
    return re.sub(r"[\r\n]+", " ", s).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--dev-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--override-lang", type=str, default="",
                   help="If set, use this language code for all rows (e.g. gug_Latn). Otherwise derive from category.")
    p.add_argument("--manifest-subdir", type=str, default="manifests",
                   help="Subdir under output-dir for TSV/WRD (default: manifests). Use e.g. manifests_gug for gug_Latn run.")
    args = p.parse_args()

    def load(csv_path):
        # (path, text, lang_code, expert)
        out = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = (row.get("directory") or "").strip()
                # v3 splits use final_corrected_reference_norm; fall back to older column names
                text = one_line((
                    row.get("final_corrected_reference_norm") or
                    row.get("corrected_reference_norm") or
                    row.get("reference_norm") or ""
                ).strip())
                # v3 splits use v2_category (Gemma); fall back to new_category (GPT-4o)
                raw = (row.get("v2_category") or row.get("new_category") or "").strip().lower()
                if "spanish" in raw or raw.startswith("spa"):
                    expert = "spanish_only"
                    lang_code = "spa_Latn"
                elif "guarani" in raw or raw.startswith("gug"):
                    expert = "guarani_only"
                    lang_code = "grn_Latn"
                elif "mixed" in raw:
                    expert = "mixed"
                    lang_code = "spa_Latn"
                elif "jeh" in raw:
                    # jehe'a is CS-Jopara; treat as mixed expert, default grn_Latn
                    expert = "mixed"
                    lang_code = "grn_Latn"
                else:
                    continue
                if args.override_lang:
                    lang_code = args.override_lang
                if path and text and Path(path).exists():
                    out.append((path, text, lang_code, expert))
        return out

    train = load(args.train_csv)
    dev = load(args.dev_csv)
    if not train or not dev:
        raise SystemExit("Need both train and dev rows.")

    manifest_dir = (args.output_dir / args.manifest_subdir).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name, rows):
        tsv = manifest_dir / (name + ".tsv")
        wrd = manifest_dir / (name + ".wrd")
        with tsv.open("w", encoding="utf-8") as ft, wrd.open("w", encoding="utf-8") as fw:
            ft.write("/\n")
            for path, text, lang_code, expert in rows:
                n = 0
                try:
                    with wave.open(path, "rb") as w:
                        n = w.getnframes()
                except Exception:
                    pass
                ft.write("{}\t{}\t{}\t{}\n".format(one_line(path), n, lang_code, expert))
                fw.write(text + "\n")

    write_split("train", train)
    write_split("dev", dev)
    write_split("train_all", train)
    for name, subset in [
        ("train_spanish", [r for r in train if r[3] == "spanish_only"]),
        ("train_guarani", [r for r in train if r[3] == "guarani_only"]),
        ("train_mixed", [r for r in train if r[3] == "mixed"]),
    ]:
        write_split(name, subset if subset else train)
    print("Wrote {} train, {} dev -> {} (lang override={})".format(
        len(train), len(dev), manifest_dir, args.override_lang or "none"))


if __name__ == "__main__":
    main()
