#!/usr/bin/env python3
"""
Rebuild test subsets and training splits using v2_category (Gemma re-verification).

Target test set sizes:
  guarani_only: 25    spanish_only: 25    mixed: 67    jehe'a: 33  (total: 150)

Algorithm:
  1. Take 145 non-EXC test rows, redistribute by v2_category.
     Result: guarani_only=20, spanish_only=33, mixed=45, jehe'a=47
  2. Downsample where over target (prefer to drop BCN rows first):
     spanish_only 33→25 (drop 8: BCN first), jehe'a 47→33 (drop 14: BCN first)
  3. Fill shortfalls from quality_ok training rows:
     guarani_only +5, mixed +22
  4. Build new training: quality_ok rows not in new test set, split 90/10 per v2_category.

Three training subsets (matching three LoRA models):
  train_spanish_only     — v2_category=spanish_only
  train_guarani_only     — CEGPA (v2_category=guarani_only) + CommonVoice anchor
  train_cs_jopara        — v2_category in {mixed, jehe'a}

Outputs:
  updated_csvs/v2_test_subsets/
    test_guarani_only.csv, test_spanish_only.csv, test_mixed.csv, test_jehea.csv
    report.txt
  updated_csvs/training_splits_v3/
    train_spanish_only.csv, dev_spanish_only.csv
    train_guarani_only.csv, dev_guarani_only.csv
    train_cs_jopara.csv, dev_cs_jopara.csv
    summary.txt

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/rebuild_test_training_v2.py
  python3 QC_plans/rebuild_test_training_v2.py --dry-run
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

CSV_IN          = Path("updated_csvs/June2026/Everything.csv")
CV_GUARANI      = Path("updated_csvs/by_quality/train_guarani_only_commonvoice.csv")
OLD_TEST_DIR    = Path("updated_csvs/2nd_update_test_subsets")
NEW_TEST_DIR    = Path("updated_csvs/v2_test_subsets")
NEW_TRAIN_DIR   = Path("updated_csvs/training_splits_v3")

OLD_TEST_FILES = {
    "guarani_only": "test_guarani_only.csv",
    "spanish_only": "test_spanish_only.csv",
    "mixed":        "test_mixed.csv",
    "jehea":        "test_jehea.csv",
}

TARGETS = {"guarani_only": 25, "spanish_only": 25, "mixed": 67, "jehe'a": 33}
SEED    = 42

# Output columns for test CSVs
TEST_COLS = [
    "directory",
    "final_corrected_reference_norm",
    "v2_category",
    "quality_comments",
    "file_id",
    "duration",
    "v2_spa_count", "v2_grn_count", "v2_jha_count", "v2_other_count",
    "original_test_subset",
    "source",
]

# Output columns for training CSVs
TRAIN_COLS = [
    "directory",
    "final_corrected_reference_norm",
    "v2_category",
    "quality_comments",
    "file_id",
    "duration",
    "v2_spa_count", "v2_grn_count", "v2_jha_count", "v2_other_count",
]


def quality_ok(q: str) -> bool:
    q = (q or "").strip()
    if q == "Good":
        return True
    if not q:
        return False
    return all(t.strip() in ("CRT-D", "CRT-A") for t in q.split(","))


def seg_num(path: str) -> str | None:
    m = re.search(r"_(\d+)\.wav", path)
    return m.group(1) if m else None


def to_row(ev_row: dict, orig_label: str = "", source: str = "from_training") -> dict:
    return {k: ev_row.get(k, "") for k in TRAIN_COLS} | {
        "original_test_subset": orig_label,
        "source": source,
    }


def split_90_10(rows: list, rng: random.Random) -> tuple[list, list]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * 0.10))
    return shuffled[n_dev:], shuffled[:n_dev]


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)

    # ── Load Everything.csv ──────────────────────────────────────────────────
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        ev_rows = list(csv.DictReader(f))
    ev_by_seg: dict[str, dict] = {}
    for r in ev_rows:
        s = seg_num(r.get("directory", ""))
        if s:
            ev_by_seg[s] = r
    print(f"Everything.csv: {len(ev_rows)} rows")

    # ── Load old test rows, attach v2 info ──────────────────────────────────
    old_test_segs: set[str] = set()
    all_old_test: list[dict] = []
    for orig_label, fname in OLD_TEST_FILES.items():
        with open(OLD_TEST_DIR / fname, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s = seg_num(r.get("directory", ""))
                if s:
                    old_test_segs.add(s)
                ev = ev_by_seg.get(s or "", {})
                all_old_test.append({
                    "_seg":       s,
                    "_orig":      orig_label,
                    "_v2_cat":    ev.get("v2_category", ""),
                    "_qc":        ev.get("quality_comments", ""),
                    "_ev":        ev,
                })
    print(f"Old test rows: {len(all_old_test)}  (unique segs: {len(old_test_segs)})")

    # ── Redistribute CLEAN (non-EXC, non-BCN) test rows by v2_category ────────
    # EXC and BCN rows are dropped from test entirely; shortfalls are filled
    # from quality_ok training rows so the new test set has zero EXC/BCN entries.
    exc_rows  = [r for r in all_old_test if r["_qc"] == "EXC"]
    bcn_rows  = [r for r in all_old_test if r["_qc"] != "EXC" and not quality_ok(r["_qc"])]
    clean     = [r for r in all_old_test if r["_qc"] != "EXC" and quality_ok(r["_qc"])]
    print(f"EXC dropped: {len(exc_rows)}, BCN dropped: {len(bcn_rows)}, clean: {len(clean)}")

    pools: dict[str, list] = defaultdict(list)
    for r in clean:
        pools[r["_v2_cat"]].append(r)

    print(f"\nClean pool sizes after v2 redistribution:")
    for cat, tgt in TARGETS.items():
        have = len(pools[cat])
        print(f"  {cat:<22}: {have:3}  (target {tgt}, delta {tgt-have:+d})")

    # ── Build new test sets ──────────────────────────────────────────────────
    new_test_segs: set[str] = set()
    new_test_pools: dict[str, list[dict]] = {}

    for cat, tgt in TARGETS.items():
        pool = list(pools[cat])

        if len(pool) >= tgt:
            # Downsample randomly (all are clean, no preference needed)
            rng.shuffle(pool)
            selected = pool[:tgt]
        else:
            # Start with all clean pool rows, fill from quality_ok training rows
            selected = pool
            need = tgt - len(selected)
            candidates = [
                r for r in ev_rows
                if r.get("v2_category", "") == cat
                and r.get("quality_comments", "").strip() != "EXC"
                and quality_ok(r.get("quality_comments", ""))
                and (r.get("final_corrected_reference_norm", "") or "").strip()
                and seg_num(r.get("directory", "")) not in old_test_segs
            ]
            rng.shuffle(candidates)
            drawn = candidates[:need]
            print(f"  {cat}: drawing {len(drawn)}/{need} from training ({len(candidates)} available)")
            for ev in drawn:
                s = seg_num(ev.get("directory", ""))
                selected.append({"_seg": s, "_orig": "from_training", "_v2_cat": cat,
                                  "_qc": ev.get("quality_comments", ""), "_ev": ev})

        for r in selected:
            if r["_seg"]:
                new_test_segs.add(r["_seg"])
        new_test_pools[cat] = selected

    # ── Build test CSV rows ──────────────────────────────────────────────────
    def make_test_row(r: dict) -> dict:
        ev = r["_ev"]
        return {
            "directory":                      ev.get("directory", ""),
            "final_corrected_reference_norm": ev.get("final_corrected_reference_norm", ""),
            "v2_category":                    ev.get("v2_category", ""),
            "quality_comments":               ev.get("quality_comments", ""),
            "file_id":                        ev.get("file_id", ""),
            "duration":                       ev.get("duration", ""),
            "v2_spa_count":                   ev.get("v2_spa_count", ""),
            "v2_grn_count":                   ev.get("v2_grn_count", ""),
            "v2_jha_count":                   ev.get("v2_jha_count", ""),
            "v2_other_count":                 ev.get("v2_other_count", ""),
            "original_test_subset":           r["_orig"],
            "source": "original_test" if r["_orig"] != "from_training" else "from_training",
        }

    # ── Build training ───────────────────────────────────────────────────────
    train_eligible = [
        r for r in ev_rows
        if r.get("quality_comments", "").strip() != "EXC"
        and quality_ok(r.get("quality_comments", ""))
        and (r.get("final_corrected_reference_norm", "") or "").strip()
        and seg_num(r.get("directory", "")) not in new_test_segs
    ]
    print(f"\nNew training eligible: {len(train_eligible)}")

    # Partition by subset
    grn_cegpa = [r for r in train_eligible if r.get("v2_category", "") == "guarani_only"]
    spa_rows  = [r for r in train_eligible if r.get("v2_category", "") == "spanish_only"]
    csj_rows  = [r for r in train_eligible if r.get("v2_category", "") in ("mixed", "jehe'a")]

    print(f"\nTraining subset counts (before 90/10 split):")
    print(f"  spanish_only  : {len(spa_rows)}")
    print(f"  cs_jopara     : {len(csj_rows)}  (mixed+jehe'a)")
    print(f"  guarani_only  : {len(grn_cegpa)}  CEGPA + CommonVoice")

    spa_train, spa_dev   = split_90_10(spa_rows, rng)
    csj_train, csj_dev   = split_90_10(csj_rows, rng)
    grn_train_cegpa, grn_dev_cegpa = split_90_10(grn_cegpa, random.Random(SEED + 1))

    # Add CommonVoice
    if CV_GUARANI.exists():
        with open(CV_GUARANI, newline="", encoding="utf-8") as f:
            cv_rows = list(csv.DictReader(f))
        print(f"  CommonVoice Guaraní: {len(cv_rows)} rows")
        cv_out = []
        for r in cv_rows:
            tx = (r.get("corrected_reference_norm", "") or r.get("reference_norm", "") or "").strip()
            cv_out.append({
                "directory": r.get("directory", ""),
                "final_corrected_reference_norm": tx,
                "v2_category": "guarani_only",
                "quality_comments": r.get("quality_comments", "") or "Good",
                "file_id": "CV", "duration": r.get("duration", ""),
                "v2_spa_count": "", "v2_grn_count": "", "v2_jha_count": "", "v2_other_count": "",
            })
        cv_train, cv_dev = split_90_10(cv_out, random.Random(SEED + 2))
        grn_train = cv_train + grn_train_cegpa
        grn_dev   = cv_dev
    else:
        print(f"  WARNING: {CV_GUARANI} not found")
        grn_train = grn_train_cegpa
        grn_dev   = grn_dev_cegpa

    print(f"\nFinal training/dev split:")
    print(f"  spanish_only train={len(spa_train)} dev={len(spa_dev)}")
    print(f"  cs_jopara    train={len(csj_train)} dev={len(csj_dev)}")
    print(f"  guarani_only train={len(grn_train)} dev={len(grn_dev)}")

    if args.dry_run:
        # Validate test sizes
        print(f"\nNew test set sizes (dry-run validation):")
        for cat, rows in new_test_pools.items():
            src = {r["_orig"] for r in rows}
            print(f"  {cat:<22}: {len(rows)}  sources={src}")
        print("\nDry-run — no files written.")
        return

    # ── Write test subsets ───────────────────────────────────────────────────
    test_file_map = {
        "guarani_only": "test_guarani_only.csv",
        "spanish_only": "test_spanish_only.csv",
        "mixed":        "test_mixed.csv",
        "jehe'a":       "test_jehea.csv",
    }
    for cat, fname in test_file_map.items():
        out_rows = [make_test_row(r) for r in new_test_pools[cat]]
        write_csv(NEW_TEST_DIR / fname, out_rows, TEST_COLS)
        from_training = sum(1 for r in new_test_pools[cat] if r["_orig"] == "from_training")
        print(f"Wrote {NEW_TEST_DIR}/{fname}: {len(out_rows)} rows "
              f"(+{from_training} from training, 0 BCN/EXC)")

    # ── Write training splits ────────────────────────────────────────────────
    write_csv(NEW_TRAIN_DIR / "train_spanish_only.csv", spa_train, TRAIN_COLS)
    write_csv(NEW_TRAIN_DIR / "dev_spanish_only.csv",   spa_dev,   TRAIN_COLS)
    write_csv(NEW_TRAIN_DIR / "train_cs_jopara.csv",    csj_train, TRAIN_COLS)
    write_csv(NEW_TRAIN_DIR / "dev_cs_jopara.csv",      csj_dev,   TRAIN_COLS)
    write_csv(NEW_TRAIN_DIR / "train_guarani_only.csv", grn_train, TRAIN_COLS)
    write_csv(NEW_TRAIN_DIR / "dev_guarani_only.csv",   grn_dev,   TRAIN_COLS)
    print(f"\nTraining splits written to {NEW_TRAIN_DIR}/")

    # ── Write summary ────────────────────────────────────────────────────────
    report = [
        "v2 test subsets rebuild — report",
        "=" * 60, "",
        f"EXC rows dropped from test: {len(exc_rows)}",
        "  " + ", ".join(f"{r['_ev'].get('directory','').split('/')[-1]}" for r in exc_rows),
        f"BCN rows dropped from test: {len(bcn_rows)}",
        "",
        "New test set composition (all rows are quality_ok — no EXC, no BCN):",
    ]
    for cat, fname in test_file_map.items():
        rows = new_test_pools[cat]
        orig_src: dict[str, int] = {}
        for r in rows:
            orig_src[r["_orig"]] = orig_src.get(r["_orig"], 0) + 1
        report.append(f"  {cat:<22}: {len(rows)}  sources={orig_src}")
    report += [
        "",
        "Training split summary:",
        f"  spanish_only  train={len(spa_train)} dev={len(spa_dev)}",
        f"  cs_jopara     train={len(csj_train)} dev={len(csj_dev)}",
        f"  guarani_only  train={len(grn_train)} dev={len(grn_dev)}",
        "",
        "Models and their test sets:",
        "  Spanish-only model  → test_spanish_only.csv (25 rows)",
        "  Guaraní-only model  → test_guarani_only.csv (25 rows)",
        "  CS-Jopara model     → test_mixed.csv (67) + test_jehea.csv (33) = 100 rows",
    ]
    (NEW_TEST_DIR / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nReport: {NEW_TEST_DIR}/report.txt")


if __name__ == "__main__":
    main()
