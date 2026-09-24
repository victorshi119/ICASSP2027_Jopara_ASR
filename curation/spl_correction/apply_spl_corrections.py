#!/usr/bin/env python3
"""
Phase 2: Apply SPL corrections to Everything.csv using spl_vocab_bank.csv.

Steps:
  1. Vocab bank pass  — exact case-insensitive token match on corrected_reference_norm.
  2. Second-pass ReaLLM sweep — send unmatched rows to gpt-oss-120b to catch stragglers.
  3. Write three NEW columns (spl_tokens_found, spl_corrected_text,
     final_corrected_reference_norm) to Everything.csv.
  4. Write QC_plans/spl_changes_log.csv audit trail.

Run from: /N/project/icassp2026/Jopara_ASR_models/
  python3 QC_plans/apply_spl_corrections.py
  python3 QC_plans/apply_spl_corrections.py --skip-second-pass   # vocab bank only
"""

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_IN      = Path("updated_csvs/June2026/Everything.csv")
CSV_BACKUP  = Path("updated_csvs/June2026/Everything.csv.bak2")
BANK_CSV    = Path("QC_plans/spl_vocab_bank.csv")
BANK_BACKUP = Path("QC_plans/spl_vocab_bank.csv.bak2")
CHANGES_LOG = Path("QC_plans/spl_changes_log.csv")
CACHE2      = Path("QC_plans/spl_second_pass_cache.jsonl")

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 25
SLEEP_BETWEEN_CALLS = 0.6
MODEL               = "gpt-oss-120b"

SYSTEM_PROMPT = """\
You are a linguist specializing in Paraguayan Spanish–Guaraní (Jopara) contact speech.

Your task is to identify SPL words: Spanish vocabulary words written using Guaraní \
phonological spelling conventions. SPL words are 100% Spanish in meaning but respelled \
according to Guaraní orthographic patterns.

Common Guaraní spelling conventions that produce SPL forms:
  - 'k' replaces 'c' or 'qu'  (eskuela → escuela, pórke → porque)
  - Accented vowels dropped or shifted (edukasion → educación, amígo → amigo)
  - Double letters simplified  (kavaju → caballo)
  - Phonetic vowel spelling    (exklavisante → esclavizante)
  - Word fusion                (enserio → en serio)

SPL is NOT:
  - Actual Guaraní words (e.g., oguata, upépe, ha, ndaipóri, pe, ko, che, nde, piko)
  - Jehe'a tokens: Spanish root + Guaraní morphological affix (e.g., o-entende, gana-ta,
    eskuela-pe — these have Guaraní morphology attached)
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


# ── API helpers ────────────────────────────────────────────────────────────
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


def parse_response(raw: str) -> list[tuple[str, str, str]]:
    if not raw or raw.strip().upper() == "NONE":
        return []
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
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


# ── Vocab bank ─────────────────────────────────────────────────────────────
def load_vocab_bank(bank_path: Path) -> dict[str, str]:
    """Returns {guarani_spelling.lower(): standard_spanish}."""
    vocab = {}
    with open(bank_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = row["guarani_spelling"].strip().lower()
            s = row["standard_spanish"].strip()
            if g and s:
                vocab[g] = s
    return vocab


def load_vocab_bank_full(bank_path: Path) -> dict[str, dict]:
    """Returns full dict for writing back."""
    pairs = {}
    with open(bank_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = row["guarani_spelling"].strip().lower()
            pairs[g] = {
                "standard_spanish": row["standard_spanish"].strip(),
                "source": row["source"].strip(),
                "batch_count": int(row["batch_occurrences"] or 0),
                "example_utterance": row["example_utterance_index"].strip(),
            }
    return pairs


def write_vocab_bank(pairs: dict, bank_path: Path) -> None:
    with open(bank_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["guarani_spelling", "standard_spanish", "source",
                         "batch_occurrences", "example_utterance_index"])
        for g, info in sorted(pairs.items()):
            writer.writerow([
                g, info["standard_spanish"], info["source"],
                info.get("batch_count", 0), info.get("example_utterance", ""),
            ])


# ── SPL matching ───────────────────────────────────────────────────────────
def apply_vocab_bank(text: str, vocab: dict[str, str]) -> tuple[list[str], str]:
    """
    Returns (found_original_tokens, corrected_text).
    Replaces matched tokens in-place; word-fusion replacements (enserio→en serio)
    expand naturally since we join the corrected token list.
    """
    tokens = text.split()
    found = []
    corrected_tokens = []
    for tok in tokens:
        key = tok.lower()
        if key in vocab:
            found.append(tok)
            corrected_tokens.append(vocab[key])
        else:
            corrected_tokens.append(tok)
    return found, " ".join(corrected_tokens)


# ── Second-pass cache ──────────────────────────────────────────────────────
def load_second_pass_cache(cache_path: Path) -> tuple[set[int], list[tuple[str, str, str]]]:
    """Returns (done_batch_starts, all_discovered_pairs)."""
    done: set[int] = set()
    all_pairs: list[tuple[str, str, str]] = []
    if not cache_path.exists():
        return done, all_pairs
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["row_start"])
                all_pairs.extend(tuple(x) for x in obj.get("results", []))
            except Exception:
                pass
    return done, all_pairs


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-second-pass", action="store_true",
                        help="Apply vocab bank only; skip second-pass ReaLLM sweep.")
    args = parser.parse_args()

    # Load vocab bank
    vocab      = load_vocab_bank(BANK_CSV)
    vocab_full = load_vocab_bank_full(BANK_CSV)
    print(f"Vocab bank loaded: {len(vocab)} entries")

    # Load Everything.csv
    with open(CSV_IN, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows      = list(reader)
    print(f"Loaded {len(rows)} rows from {CSV_IN}")

    for col in ("spl_tokens_found", "spl_corrected_text"):
        if col in fieldnames:
            print(f"  Note: column '{col}' already exists — will overwrite")

    # Backup
    shutil.copy2(CSV_IN, CSV_BACKUP)
    shutil.copy2(BANK_CSV, BANK_BACKUP)
    print(f"Backups written: {CSV_BACKUP.name}, {BANK_BACKUP.name}")

    # ── Pass 1: vocab bank matching ────────────────────────────────────────
    print("\n── Pass 1: Vocab bank matching ──────────────────────────────")
    # results[i] = (row_dict, spl_tokens_found_str, spl_corrected_text_str)
    results: list[tuple[dict, str, str]] = []
    pass1_matched = 0
    for row in rows:
        is_exc = (row.get("quality_comments") or "").strip() == "EXC"
        text   = (row.get("corrected_reference_norm") or "").strip()
        if is_exc or not text:
            results.append((row, "", ""))
            continue
        found, corrected = apply_vocab_bank(text, vocab)
        if found:
            pass1_matched += 1
            results.append((row, "|".join(found), corrected))
        else:
            results.append((row, "", ""))

    print(f"Pass 1 complete: {pass1_matched} rows matched")

    # ── Pass 2: second-pass ReaLLM sweep ──────────────────────────────────
    if not args.skip_second_pass:
        print("\n── Pass 2: ReaLLM sweep on unmatched rows ───────────────────")
        unmatched_indices = [
            i for i, (row, found_str, _) in enumerate(results)
            if not found_str
            and (row.get("corrected_reference_norm") or "").strip()
            and (row.get("quality_comments") or "").strip() != "EXC"
        ]
        print(f"Unmatched rows to sweep: {len(unmatched_indices)}")

        done_starts, cached_pairs = load_second_pass_cache(CACHE2)

        # Seed vocab with any forms already in the cache
        for guarani, standard, example_idx in cached_pairs:
            g = guarani.lower()
            if g not in vocab:
                vocab[g] = standard
                vocab_full[g] = {
                    "standard_spanish": standard,
                    "source": "reallm_pass2",
                    "batch_count": 1,
                    "example_utterance": example_idx,
                }

        batches   = [unmatched_indices[i:i+BATCH_SIZE]
                     for i in range(0, len(unmatched_indices), BATCH_SIZE)]
        remaining = [b for b in batches if b[0] not in done_starts]
        print(f"Batches: {len(batches)}  |  Cached: {len(done_starts)}  |  Remaining: {len(remaining)}")

        new_forms_total = 0
        client = make_client()

        all_new_pairs: list[tuple[str, str, str]] = list(cached_pairs)

        with open(CACHE2, "a", encoding="utf-8") as cache_f:
            for b_idx, batch_indices in enumerate(remaining):
                batch_rows = [(idx, results[idx][0]) for idx in batch_indices]
                lines      = [
                    f"[{idx}] {row.get('corrected_reference_norm', '').strip()}"
                    for idx, row in batch_rows
                ]
                user_msg = USER_TEMPLATE.format(utterances="\n".join(lines))

                print(f"Batch {len(done_starts)+b_idx+1}/{len(batches)} "
                      f"(result rows {batch_indices[0]}–{batch_indices[-1]})...",
                      end=" ", flush=True)

                raw    = call_with_retry(client, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ])
                parsed = parse_response(raw)
                print(f"{len(parsed)} new SPL pair(s)")

                cache_f.write(json.dumps({
                    "row_start": batch_indices[0],
                    "results":   parsed,
                    "raw_response": raw,
                }) + "\n")
                cache_f.flush()

                # Add to running vocab and pair list
                for guarani, standard, example_idx in parsed:
                    g = guarani.lower()
                    if g not in vocab:
                        vocab[g] = standard
                        vocab_full[g] = {
                            "standard_spanish": standard,
                            "source": "reallm_pass2",
                            "batch_count": 1,
                            "example_utterance": example_idx,
                        }
                        new_forms_total += 1
                    else:
                        vocab_full[g]["batch_count"] = vocab_full[g].get("batch_count", 0) + 1

                all_new_pairs.extend(parsed)
                time.sleep(SLEEP_BETWEEN_CALLS)

        # Re-apply the now-enriched vocab to ALL originally-unmatched rows
        pass2_matched = 0
        for idx in unmatched_indices:
            row  = results[idx][0]
            text = (row.get("corrected_reference_norm") or "").strip()
            found, corrected = apply_vocab_bank(text, vocab)
            if found:
                results[idx] = (row, "|".join(found), corrected)
                pass2_matched += 1

        print(f"Pass 2 complete: {new_forms_total} new forms, {pass2_matched} additional rows corrected")

        if new_forms_total > 0:
            write_vocab_bank(vocab_full, BANK_CSV)
            print(f"Vocab bank updated ({len(vocab_full)} entries): {BANK_CSV}")

    # ── Write updated Everything.csv ───────────────────────────────────────
    total_with_spl = sum(1 for _, found_str, _ in results if found_str)
    print(f"\nTotal rows with SPL corrections: {total_with_spl}")

    new_fieldnames = list(fieldnames)
    for col in ("spl_tokens_found", "spl_corrected_text", "final_corrected_reference_norm"):
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    with open(CSV_IN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row, found_str, corrected_str in results:
            row["spl_tokens_found"]   = found_str
            row["spl_corrected_text"] = corrected_str
            row["final_corrected_reference_norm"] = (
                corrected_str if corrected_str
                else (row.get("corrected_reference_norm") or "").strip()
            )
            writer.writerow(row)
    print(f"Written: {CSV_IN}")

    # ── Write changes log ──────────────────────────────────────────────────
    changes = [(row, fs, cs) for row, fs, cs in results if fs]
    with open(CHANGES_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["directory", "base_text", "spl_tokens_found", "spl_corrected_text"])
        for row, found_str, corrected_str in changes:
            writer.writerow([
                row.get("directory", ""),
                row.get("corrected_reference_norm", ""),
                found_str,
                corrected_str,
            ])
    print(f"Changes log: {CHANGES_LOG}  ({len(changes)} rows)")


if __name__ == "__main__":
    main()
