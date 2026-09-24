#!/usr/bin/env python3
"""
Task 2: Word-level LID re-verification via gemma-4-31B-it (IU ReaLLMs).

For each non-EXC utterance in Everything.csv, sends the tokenized
final_corrected_reference_norm to gemma-4-31B-it and gets back
word-level tags: s=Spanish, g=Guaraní, j=Jehe'a, o=other.

Outputs:
  1. Tagged files per interview:
       /N/slate/wencshi/JoparaASR_v2/LID_data/v2/
         omnijehea-v2-tr-NNN-tokenized_tagged.txt
     Format: #NNN#SSSSS . .  (NNN=interview, SSSSS=audio segment number)
             1\tword\ttag
             2\tword\ttag
             ...
  2. New columns appended to Everything.csv:
       v2_spa_count, v2_grn_count, v2_jha_count, v2_other_count, v2_category
  3. Resume cache: QC_plans/lid_cache.jsonl (keyed by directory value)

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/run_lid_reverification.py
  python3 QC_plans/run_lid_reverification.py --dry-run   # show counts, no API calls
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_IN      = Path("updated_csvs/June2026/Everything.csv")
CSV_BACKUP  = Path("updated_csvs/June2026/Everything.csv.bak_pre_lid_v2")
CACHE       = Path("QC_plans/lid_cache.jsonl")
LID_OUT_DIR = Path("/N/slate/wencshi/JoparaASR_v2/LID_data/v2")

# ── Config ─────────────────────────────────────────────────────────────────
SLEEP_BETWEEN_CALLS = 0.6
MODEL               = "gemma-4-31B-it"
INTERVIEWS          = ["tr-005", "tr-006", "tr-007", "tr-008", "tr-009", "tr-010", "tr-011"]

SYSTEM_PROMPT = """\
You are a precise token-level language tagger for Spanish–Guaraní (Jopara).
Return exactly one line per input token in the format: index<TAB>word<TAB>tag.
Use tags: s=Spanish, g=Guarani, j=Jopara (ONLY if Spanish root + Guarani morphology), o=other.\
"""

USER_TEMPLATE = """\
For each word in the following numbered list, assign a language label:
- 's' for Spanish,
- 'g' for Guarani,
- 'j' (Jopara/Mixed): STRICTLY for tokens mixing Spanish roots with Guarani morphology.
   - Look for Guarani subject prefixes: a-, re-, o-, ja-, ña-, ro-, pe-, i- on Spanish verbs.
   - Look for Guarani TAM suffixes: -ta, -ma, -pa, -se, -kuri, -ne on Spanish roots.
   - Look for Guarani case markers: -pe, -gui, -re on Spanish nouns.
   - Example: "o-entende" (j), "eskuela-pe" (j), "gana-ta" (j).
- 'o' for other/special (e.g., punctuation, names, interjections, inaudible).

Output exactly as: index<TAB>word<TAB>tag
Do NOT output any other text, headers, or explanations.

Here is the list:
{unit_text}
"""

VALID_TAGS = {"s", "g", "j", "o"}

# ── Filler normalization ────────────────────────────────────────────────────
# Paralinguistic tokens (hesitations, laughter, backchannels) are language-neutral
# and must always be tagged 'o', regardless of what the model returns.
FILLERS = {
    "eh","ehh","ehhh","eeh","eeeh","eehhh","ehhm","ehem",
    "ah","ahh","ahhh","aah","aaah",
    "oh","ohh",
    "uh","uhh",
    "em","emm","um","umm",
    "mm","mmm","mmmm","mmh","mmmh","mhm","mmhm","mhmh",
    "hm","hmm","hmmm","hhmmm","mh","mhh",
    "jaja","jajaja","jajajaja",
    "jeje","jejeje","jejejeje",
    "jiji",
    "haha","hahaha",
    "hehe",
    "aha",
}
# Guaraní words that look like fillers — do NOT override
GUARANI_KEEP = {"jaha", "haja", "aja", "ajá"}


def normalize_filler(tok: str, tag: str) -> str:
    """Return 'o' if tok is a paralinguistic filler, else return tag unchanged."""
    t = tok.lower().strip(".,?!;:()")
    if t in GUARANI_KEEP:
        return tag
    return "o" if t in FILLERS else tag


# ── API helpers ────────────────────────────────────────────────────────────
def load_api_key() -> str:
    # Prefer env var if already set (e.g. sourced by sbatch script)
    if os.environ.get("REALLMS_API_KEY"):
        return os.environ["REALLMS_API_KEY"]
    secrets = Path.home() / ".reallms_secrets"
    with open(secrets) as f:
        for line in f:
            if "REALLMS_API_KEY" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("REALLMS_API_KEY not found in env or ~/.reallms_secrets")


def make_client() -> OpenAI:
    return OpenAI(
        api_key=load_api_key(),
        base_url="https://reallms.rescloud.iu.edu/direct/v1",
    )


def call_with_retry(client: OpenAI, messages: list, max_retries: int = 8) -> str:
    base_wait, max_wait = 1.5, 40.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            transient = any(x in msg for x in
                            ("rate limit", "429", "timeout", "temporar", "server", "overload"))
            if not transient or attempt == max_retries - 1:
                raise
            wait = min(max_wait, base_wait * (2 ** attempt)) * (0.85 + 0.30 * random.random())
            print(f"    [retry {attempt+1}/{max_retries}] sleeping {wait:.1f}s — {e}")
            time.sleep(wait)
    raise RuntimeError("Exceeded max retries")


# ── Tag parsing ────────────────────────────────────────────────────────────
def parse_tag_response(raw: str, tokens: list[str]) -> list[str]:
    """
    Parse gemma's line-by-line tag output. Returns one tag per input token.
    Falls back to 'o' for tokens the model dropped or mis-formatted.
    """
    tags_by_idx: dict[int, str] = {}

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept tab or multiple-space separation
        parts = re.split(r"\t| {2,}", line, maxsplit=2)
        if len(parts) < 2:
            # Try single-space split as last resort
            parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # Tag is the last non-empty part
        tag_raw = parts[-1].strip().lower().rstrip(".,;:")
        tag = tag_raw if tag_raw in VALID_TAGS else "o"
        tags_by_idx[idx] = tag

    # Align to input tokens (1-based index) and apply filler normalization
    result = []
    for i, tok in enumerate(tokens, start=1):
        raw_tag = tags_by_idx.get(i, "o")
        result.append(normalize_filler(tok, raw_tag))
    return result


# ── Counts and category ────────────────────────────────────────────────────
def compute_counts(tags: list[str]) -> tuple[int, int, int, int]:
    """Returns (spa, grn, jha, other)."""
    return (
        tags.count("s"),
        tags.count("g"),
        tags.count("j"),
        tags.count("o"),
    )


def compute_category(spa: int, grn: int, jha: int) -> str:
    if jha > 0:
        return "jehe'a"
    if spa > 0 and grn == 0:
        return "spanish_only"
    if grn > 0 and spa == 0:
        return "guarani_only"
    if spa > 0 and grn > 0:
        return "mixed"
    return "unknown"


# ── Cache ──────────────────────────────────────────────────────────────────
def load_cache(cache_path: Path) -> dict[str, dict]:
    """Returns {directory: {tags, spa, grn, jha, other, category}}."""
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cache[obj["directory"]] = obj
            except Exception:
                pass
    return cache


def append_cache(cache_path: Path, entry: dict) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show row counts per interview; no API calls.")
    args = parser.parse_args()

    # Load Everything.csv
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows      = list(reader)
    print(f"Loaded {len(rows)} rows from {CSV_IN}")

    # Group rows by interview, sorted by audio segment number
    def seg_num(row: dict) -> int:
        m = re.search(r"_(\d+)\.wav", row.get("directory", ""))
        return int(m.group(1)) if m else 0

    interview_rows: dict[str, list[dict]] = {iv: [] for iv in INTERVIEWS}
    for row in rows:
        iv = row.get("file_id", "")
        if iv in interview_rows:
            interview_rows[iv].append(row)
    for iv in INTERVIEWS:
        interview_rows[iv].sort(key=seg_num)

    # Count processable rows
    total_process = sum(
        1 for iv in INTERVIEWS for r in interview_rows[iv]
        if (r.get("final_corrected_reference_norm") or "").strip()
    )
    total_skip = sum(
        1 for iv in INTERVIEWS for r in interview_rows[iv]
        if not (r.get("final_corrected_reference_norm") or "").strip()
    )
    print(f"Rows to tag: {total_process}  |  EXC/empty (skip): {total_skip}")
    for iv in INTERVIEWS:
        iv_rows = interview_rows[iv]
        n_ok = sum(1 for r in iv_rows if (r.get("final_corrected_reference_norm") or "").strip())
        print(f"  {iv}: {n_ok}/{len(iv_rows)} processable")

    if args.dry_run:
        print("Dry-run mode — exiting without API calls.")
        return

    # Load cache
    cache = load_cache(CACHE)
    cached_count = sum(1 for iv in INTERVIEWS for r in interview_rows[iv]
                       if r.get("directory", "") in cache)
    print(f"Cache: {len(cache)} entries  ({cached_count} of {total_process} already done)")

    # Create output directory
    LID_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a lookup from directory → row index in `rows` for writing results back
    dir_to_row_idx: dict[str, int] = {r.get("directory", ""): i for i, r in enumerate(rows)}

    client = make_client()
    done = 0
    errors = 0

    for iv in INTERVIEWS:
        iv_num = iv.replace("tr-", "")  # "005", "006", etc.
        out_path = LID_OUT_DIR / f"omnijehea-v2-{iv}-tokenized_tagged.txt"
        iv_rows  = interview_rows[iv]

        print(f"\n── {iv} ({len(iv_rows)} rows) → {out_path.name} ──────────")

        with open(out_path, "w", encoding="utf-8") as out_f:
            for row in iv_rows:
                directory = row.get("directory", "")
                text = (row.get("final_corrected_reference_norm") or "").strip()
                seg_n = re.search(r"_(\d+)\.wav", directory)
                seg_id = seg_n.group(1) if seg_n else "00000"
                header = f"#{iv_num}#{seg_id} . ."

                if not text:
                    # EXC or empty — write no block; leave new columns empty
                    continue

                # Check cache first
                if directory in cache:
                    entry = cache[directory]
                    tags  = entry["tags"]
                    tokens = text.split()
                    # Pad/trim tags if cached entry mismatches (shouldn't happen)
                    while len(tags) < len(tokens):
                        tags.append("o")
                    tags = tags[:len(tokens)]
                else:
                    tokens = text.split()
                    unit_text = "\n".join(f"{i}\t{w}" for i, w in enumerate(tokens, 1))
                    user_msg  = USER_TEMPLATE.format(unit_text=unit_text)

                    try:
                        raw  = call_with_retry(client, [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg},
                        ])
                        tags = parse_tag_response(raw, tokens)
                    except Exception as e:
                        print(f"  ERROR {directory}: {e} — tagging all as 'o'")
                        tags = ["o"] * len(tokens)
                        errors += 1

                    spa, grn, jha, other = compute_counts(tags)
                    entry = {
                        "directory": directory,
                        "tags":      tags,
                        "tokens":    tokens,
                        "spa":       spa,
                        "grn":       grn,
                        "jha":       jha,
                        "other":     other,
                        "category":  compute_category(spa, grn, jha),
                    }
                    cache[directory] = entry
                    append_cache(CACHE, entry)
                    time.sleep(SLEEP_BETWEEN_CALLS)

                # Write tagged block to file
                out_f.write(f"{header}\n")
                tokens = text.split()
                for i, (tok, tag) in enumerate(zip(tokens, tags), 1):
                    out_f.write(f"{i}\t{tok}\t{tag}\n")
                out_f.write("\n")

                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total_process} done ({errors} errors)")

        print(f"  Wrote {out_path}")

    print(f"\nTagging complete: {done} utterances, {errors} errors")

    # ── Write new columns to Everything.csv ───────────────────────────────
    print("\nWriting new columns to Everything.csv ...")
    shutil.copy2(CSV_IN, CSV_BACKUP)
    print(f"  Backup: {CSV_BACKUP}")

    new_cols = ["v2_spa_count", "v2_grn_count", "v2_jha_count", "v2_other_count", "v2_category"]
    new_fieldnames = list(fieldnames)
    for col in new_cols:
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    with open(CSV_IN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            directory = row.get("directory", "")
            if directory in cache:
                entry = cache[directory]
                row["v2_spa_count"]  = entry["spa"]
                row["v2_grn_count"]  = entry["grn"]
                row["v2_jha_count"]  = entry["jha"]
                row["v2_other_count"] = entry["other"]
                row["v2_category"]   = entry["category"]
            else:
                for col in new_cols:
                    row[col] = ""
            writer.writerow(row)

    tagged = sum(1 for r in rows if r.get("directory", "") in cache)
    print(f"  {tagged} rows tagged; {len(rows) - tagged} left empty (EXC/empty text)")
    print(f"Written: {CSV_IN}")


if __name__ == "__main__":
    main()
