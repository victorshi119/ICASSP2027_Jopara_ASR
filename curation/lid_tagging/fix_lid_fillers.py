#!/usr/bin/env python3
"""
Post-hoc filler correction for Task 2 LID tags.

Paralinguistic tokens (hesitations, laughter, backchannels) are language-neutral
and must be tagged 'o', not 's' or 'g'. The initial gemma-4-31B-it pass
mis-tagged 157 such tokens across 2,999 utterances.

This script:
  1. Re-tags all filler tokens to 'o' in the JSONL cache.
  2. Rewrites the 7 tagged output files.
  3. Recomputes v2_spa_count, v2_grn_count, v2_jha_count, v2_other_count, v2_category
     and updates Everything.csv.

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/fix_lid_fillers.py
"""

import csv, json, re, shutil
from collections import Counter
from pathlib import Path

CSV_IN      = Path("updated_csvs/June2026/Everything.csv")
CSV_BACKUP  = Path("updated_csvs/June2026/Everything.csv.bak_pre_filler_fix")
CACHE       = Path("QC_plans/lid_cache.jsonl")
CACHE_BAK   = Path("QC_plans/lid_cache.jsonl.bak_pre_filler_fix")
LID_OUT_DIR = Path("/N/slate/wencshi/JoparaASR_v2/LID_data/v2")
INTERVIEWS  = ["tr-005", "tr-006", "tr-007", "tr-008", "tr-009", "tr-010", "tr-011"]

# ── Filler definition ──────────────────────────────────────────────────────
# Tokens that are unambiguously paralinguistic (hesitations, laughter, backchannels).
# These are language-neutral and must be tagged 'o'.
FILLERS = {
    # hesitation vowels / prolonged sounds
    "eh","ehh","ehhh","eeh","eeeh","eehhh","ehhm","ehem",
    "ah","ahh","ahhh","aah","aaah","ahhh",
    "oh","ohh",
    "uh","uhh",
    "em","emm","um","umm",
    # nasal hesitations
    "mm","mmm","mmmm","mmh","mmmh","mhm","mmhm","mhmh",
    "hm","hmm","hmmm","hhmmm","mh","mhh","mmh",
    # laughter (only unambiguous repeated syllable patterns)
    "jaja","jajaja","jajajaja",
    "jeje","jejeje","jejejeje",
    "jiji",
    "haha","hahaha",
    "hehe",
    # affirmative backchannel
    "aha",
}
# Guaraní words that superficially resemble fillers — do NOT override
GUARANI_KEEP = {"jaha", "haja", "aja", "ajá"}


def is_filler(tok: str) -> bool:
    t = tok.lower().strip(".,?!;:()")
    if t in GUARANI_KEEP:
        return False
    return t in FILLERS


# ── Category logic ─────────────────────────────────────────────────────────
def compute_cat(spa: int, grn: int, jha: int) -> str:
    if jha > 0:              return "jehe'a"
    if spa > 0 and grn == 0: return "spanish_only"
    if grn > 0 and spa == 0: return "guarani_only"
    if spa > 0 and grn > 0:  return "mixed"
    return "unknown"


# ── Load and fix cache ─────────────────────────────────────────────────────
def fix_cache() -> dict[str, dict]:
    shutil.copy2(CACHE, CACHE_BAK)
    print(f"Cache backup: {CACHE_BAK}")

    cache: dict[str, dict] = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cache[obj["directory"]] = obj

    total_fixed = 0
    cat_changes = []

    for directory, entry in cache.items():
        old_tags = list(entry["tags"])
        new_tags = ["o" if is_filler(tok) else tag
                    for tok, tag in zip(entry["tokens"], old_tags)]

        if new_tags != old_tags:
            fixed_count = sum(1 for o, n in zip(old_tags, new_tags) if o != n)
            total_fixed += fixed_count

            spa = new_tags.count("s")
            grn = new_tags.count("g")
            jha = new_tags.count("j")
            other = new_tags.count("o")
            new_cat = compute_cat(spa, grn, jha)
            old_cat = entry["category"]

            entry["tags"]     = new_tags
            entry["spa"]      = spa
            entry["grn"]      = grn
            entry["jha"]      = jha
            entry["other"]    = other
            entry["category"] = new_cat

            if new_cat != old_cat:
                changed_toks = [(tok, old, new)
                                for tok, old, new in zip(entry["tokens"], old_tags, new_tags)
                                if old != new]
                cat_changes.append((directory.split("/")[-1], old_cat, new_cat, changed_toks))

    print(f"Tokens re-tagged to 'o': {total_fixed}")
    print(f"Utterances with category change: {len(cat_changes)}")
    for name, old, new, toks in cat_changes:
        print(f"  {name}: {old} → {new}  (fixed: {toks})")

    # Rewrite cache
    with open(CACHE, "w", encoding="utf-8") as f:
        for entry in cache.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Cache rewritten: {CACHE}")

    return cache


# ── Rewrite tagged output files ────────────────────────────────────────────
def rewrite_tagged_files(cache: dict[str, dict]) -> None:
    import csv as _csv

    # Build directory → entry lookup
    dir_lookup = cache  # already keyed by directory path

    # Group cache entries by interview
    iv_entries: dict[str, list] = {iv: [] for iv in INTERVIEWS}
    for entry in dir_lookup.values():
        iv = re.search(r"(tr-\d+)_", entry["directory"])
        if iv and iv.group(1) in iv_entries:
            iv_entries[iv.group(1)].append(entry)

    for iv in INTERVIEWS:
        entries = sorted(iv_entries[iv],
                         key=lambda e: int(re.search(r"_(\d+)\.wav", e["directory"]).group(1)))
        iv_num  = iv.replace("tr-", "")
        out_path = LID_OUT_DIR / f"omnijehea-v2-{iv}-tokenized_tagged.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            for entry in entries:
                seg_m  = re.search(r"_(\d+)\.wav", entry["directory"])
                seg_id = seg_m.group(1) if seg_m else "00000"
                f.write(f"#{iv_num}#{seg_id} . .\n")
                for i, (tok, tag) in enumerate(zip(entry["tokens"], entry["tags"]), 1):
                    f.write(f"{i}\t{tok}\t{tag}\n")
                f.write("\n")

        print(f"  Rewrote {out_path.name} ({len(entries)} utterances)")


# ── Update Everything.csv ──────────────────────────────────────────────────
def update_csv(cache: dict[str, dict]) -> None:
    shutil.copy2(CSV_IN, CSV_BACKUP)
    print(f"CSV backup: {CSV_BACKUP}")

    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows       = list(reader)

    new_cols = ["v2_spa_count", "v2_grn_count", "v2_jha_count", "v2_other_count", "v2_category"]

    with open(CSV_IN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            entry = cache.get(row.get("directory", ""))
            if entry:
                row["v2_spa_count"]  = entry["spa"]
                row["v2_grn_count"]  = entry["grn"]
                row["v2_jha_count"]  = entry["jha"]
                row["v2_other_count"] = entry["other"]
                row["v2_category"]   = entry["category"]
            writer.writerow(row)

    # Print new distribution
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        updated_rows = list(csv.DictReader(f))
    cats = Counter(r.get("v2_category", "") for r in updated_rows)
    print("Updated v2_category distribution:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k or '(empty)':<20s}: {v}")

    print(f"Written: {CSV_IN}")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Step 1: Fix cache ===")
    cache = fix_cache()

    print("\n=== Step 2: Rewrite tagged files ===")
    rewrite_tagged_files(cache)

    print("\n=== Step 3: Update Everything.csv ===")
    update_csv(cache)

    print("\nDone.")


if __name__ == "__main__":
    main()
