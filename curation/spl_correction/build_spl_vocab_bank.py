#!/usr/bin/env python3
"""
Phase 1: Build SPL Vocabulary Bank using gpt-oss-120b via IU ReaLLMs.

Scans all rows in Everything.csv that have a non-empty corrected_reference_norm
(EXC rows and empty corrected_reference_norm rows are skipped).

Outputs:
  QC_plans/spl_vocab_bank.csv         — unique SPL forms with standard Spanish equivalents
  QC_plans/spl_vocab_bank_cache.jsonl — per-batch raw results for resume support

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/build_spl_vocab_bank.py
  python3 QC_plans/build_spl_vocab_bank.py --dry-run   # preview first batch only
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_IN     = Path("updated_csvs/June2026/Everything.csv")
OUT_DIR    = Path("QC_plans")
BANK_OUT   = OUT_DIR / "spl_vocab_bank.csv"
CACHE_FILE = OUT_DIR / "spl_vocab_bank_cache.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 25
SLEEP_BETWEEN_CALLS = 0.6
MODEL               = "gpt-oss-120b"

# Known seed forms from paper normalization table — always included in bank
SEED_FORMS = [
    ("eskuela",       "escuela"),
    ("kavaju",        "caballo"),
    ("pórke",         "porque / por qué"),
    ("porke",         "porque / por qué"),
    ("edukasion",     "educación"),
    ("ekree",         "cree"),
    ("enserio",       "en serio"),
    ("exklavisante",  "esclavizante"),
]

# ── Prompts ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a linguist specializing in Paraguayan Spanish–Guaraní (Jopara) contact speech.

Your task is to identify SPL words: Spanish vocabulary words written using Guaraní \
phonological spelling conventions. SPL words are 100% Spanish in meaning but respelled \
according to Guaraní orthographic patterns.

Common Guaraní spelling conventions that produce SPL forms:
  - 'k' replaces 'c' or 'qu'  (eskuela → escuela, pórke → porque)
  - Accented vowels dropped    (edukasion → educación)
  - Double letters simplified  (kavaju → caballo)
  - Phonetic vowel spelling    (exklavisante → esclavizante)
  - Word fusion                (enserio → en serio)

SPL is NOT:
  - Actual Guaraní words (e.g., oguata, upépe, ha, ndaipóri, pe, ko, che, nde)
  - Jehe'a tokens: Spanish root + Guaraní morphological affix (e.g., o-entende, gana-ta,
    eskuela-pe, nomás → these have Guaraní morphology attached)
  - Standard correctly-spelled Spanish words
  - Proper nouns or names

Return ONLY a pipe-separated table, one row per unique SPL form found, NO header line:
  guarani_form | standard_spanish | example_utterance_index

Where example_utterance_index is the [N] label of one utterance where the form appears.
If no SPL words are found in the entire batch, output exactly the word: NONE
"""

USER_TEMPLATE = """\
In the following Jopara utterances, identify every SPL word.

{utterances}
"""

# ── API ────────────────────────────────────────────────────────────────────
def load_api_key() -> str:
    secrets = Path.home() / ".reallms_secrets"
    with open(secrets) as f:
        for line in f:
            if "REALLMS_API_KEY" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("REALLMS_API_KEY not found in ~/.reallms_secrets")


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
                model=MODEL,
                messages=messages,
                temperature=0,
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


# ── Parsing ────────────────────────────────────────────────────────────────
def parse_response(raw: str) -> list[tuple[str, str, str]]:
    """Returns list of (guarani_form, standard_spanish, example_idx)."""
    if not raw or raw.strip().upper() == "NONE":
        return []
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        # Accept lines that are pipe-separated or have at least one '|'
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        guarani  = parts[0].lower().strip("- *`")
        standard = parts[1].strip("- *`")
        example  = parts[2].strip() if len(parts) > 2 else ""
        if guarani and standard and guarani != standard.lower():
            results.append((guarani, standard, example))
    return results


# ── Data loading ───────────────────────────────────────────────────────────
def load_rows(csv_path: Path) -> list[tuple[int, str, str]]:
    """
    Returns (csv_row_index, filename, corrected_reference_norm) for every row
    that has a non-empty corrected_reference_norm and is not EXC.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if (row.get("quality_comments") or "").strip() == "EXC":
                continue
            text = (row.get("corrected_reference_norm") or "").strip()
            if not text:
                continue
            fname = Path(row.get("directory", "")).name
            rows.append((i, fname, text))
    return rows


def load_cache(cache_path: Path) -> tuple[set[int], dict[str, dict]]:
    """Returns (done_batch_starts, accumulated_pairs)."""
    done = set()
    pairs: dict[str, dict] = {}
    if not cache_path.exists():
        return done, pairs
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["batch_start"])
                for guarani, standard, example_idx in obj.get("results", []):
                    g = guarani.lower()
                    if g not in pairs:
                        pairs[g] = {
                            "standard_spanish": standard,
                            "source": "reallm",
                            "batch_count": 0,
                            "example_utterance": example_idx,
                        }
                    pairs[g]["batch_count"] += 1
            except Exception:
                pass
    return done, pairs


# ── Vocab bank writer ──────────────────────────────────────────────────────
def write_vocab_bank(pairs: dict[str, dict], bank_path: Path) -> None:
    with open(bank_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "guarani_spelling", "standard_spanish", "source",
            "batch_occurrences", "example_utterance_index"
        ])
        for g, info in sorted(pairs.items()):
            writer.writerow([
                g,
                info["standard_spanish"],
                info["source"],
                info.get("batch_count", 0),
                info.get("example_utterance", ""),
            ])


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first batch prompt and model response, then exit.")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    client = make_client()

    all_rows = load_rows(CSV_IN)
    print(f"Rows to process: {len(all_rows)}  (EXC and empty corrected_reference_norm excluded)")

    done_starts, pairs = load_cache(CACHE_FILE)

    # Seed known forms
    for g, s in SEED_FORMS:
        gk = g.lower()
        if gk not in pairs:
            pairs[gk] = {
                "standard_spanish": s,
                "source": "known",
                "batch_count": 0,
                "example_utterance": "seed",
            }
        else:
            pairs[gk]["source"] = "known+reallm"

    batches = [all_rows[i:i + BATCH_SIZE] for i in range(0, len(all_rows), BATCH_SIZE)]
    remaining = [b for b in batches if b[0][0] not in done_starts]
    print(f"Total batches: {len(batches)}  |  Already cached: {len(done_starts)}  "
          f"|  Remaining: {len(remaining)}")

    if args.dry_run:
        batch = remaining[0] if remaining else batches[0]
        lines = [f"[{r_i}] {text}" for r_i, _, text in batch]
        user_msg = USER_TEMPLATE.format(utterances="\n".join(lines))
        print("\n── SYSTEM PROMPT ─────────────────────────────────────")
        print(SYSTEM_PROMPT)
        print("\n── USER MESSAGE (first batch) ────────────────────────")
        print(user_msg)
        print("\n── CALLING MODEL ─────────────────────────────────────")
        raw = call_with_retry(client, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        print("RAW RESPONSE:")
        print(raw)
        parsed = parse_response(raw)
        print(f"\nParsed {len(parsed)} SPL pairs:")
        for g, s, ex in parsed:
            print(f"  {g!r:25} → {s!r}  (seen in {ex})")
        return

    with open(CACHE_FILE, "a", encoding="utf-8") as cache_f:
        for b_idx, batch in enumerate(remaining):
            batch_start = batch[0][0]
            lines   = [f"[{r_i}] {text}" for r_i, _, text in batch]
            user_msg = USER_TEMPLATE.format(utterances="\n".join(lines))

            print(f"Batch {len(done_starts)+b_idx+1}/{len(batches)} "
                  f"(rows {batch_start}–{batch[-1][0]})...", end=" ", flush=True)

            raw    = call_with_retry(client, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ])
            parsed = parse_response(raw)
            print(f"{len(parsed)} SPL pair(s) found")

            # Persist batch result
            cache_f.write(json.dumps({
                "batch_start": batch_start,
                "results":     parsed,
                "raw_response": raw,
            }) + "\n")
            cache_f.flush()

            # Merge into running pairs dict
            for guarani, standard, example_idx in parsed:
                g = guarani.lower()
                if g not in pairs:
                    pairs[g] = {
                        "standard_spanish": standard,
                        "source": "reallm",
                        "batch_count": 1,
                        "example_utterance": example_idx,
                    }
                else:
                    pairs[g]["batch_count"] += 1
                    if pairs[g]["source"] == "known":
                        pairs[g]["source"] = "known+reallm"

            time.sleep(SLEEP_BETWEEN_CALLS)

    write_vocab_bank(pairs, BANK_OUT)
    total_reallm = sum(1 for v in pairs.values() if "reallm" in v["source"])
    print(f"\nDone. Vocab bank: {BANK_OUT}")
    print(f"  Seed forms (known):     {len(SEED_FORMS)}")
    print(f"  Found by ReaLLM:        {total_reallm}")
    print(f"  Total unique SPL forms: {len(pairs)}")


if __name__ == "__main__":
    main()
