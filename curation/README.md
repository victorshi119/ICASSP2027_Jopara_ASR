# Curation

These steps turn raw CEGPA recordings into the verified benchmark (paper §2). Run them in this order.

## 1. Audio preparation (`preprocessing/`)

The seven CEGPA sessions come as 12,099 short, time-stamped segments. The steps below merge them
into 3,288 candidate utterances (6.5 hours, median 4.3 s):

| Step | What it does | Result |
|---|---|---|
| 1 `step1_merge_short_segments.py` | Merge segments shorter than 2 s with the following ones in the same session (target 2–50 s) | 12,099 → 6,635 |
| 2 `step2_merge_overlapping_segments.py` | Merge segments whose time spans overlap | 6,635 → 3,288 |
| 3 `step3_cut_segments.py` | Cut each utterance out of its session recording | 3,288 clips |
| 4 `step4_resample_16k.py` | Convert to 16 kHz, mono, 16-bit | 3,288 clips |

**Step 1 was recovered from our project notes.** Before merging, the 12,099 segments were split
at random, row by row, into a train file (8,590) and a validation file (3,509), and each file was
merged separately. Because merging ignores the speaker, some merged spans include silence or
overlapping turns, which step 2 resolves. The 6,635-row table is available on request. Paths in
step 1 are our original cluster paths.

Steps 2–4 were run in Colab notebooks. The scripts here are verbatim extracts of those cells,
and each file's header names the notebook and cells it came from. Paths are still the original
Colab paths.

## 2. Manual verification

Three native speakers listened to all 3,288 utterances and gave each one a quality code (see
`data/README.md`). This was done by hand in a shared spreadsheet, so there is no code for it. The
result is the `quality_codes` and `flags` columns in `data/metadata/cegpa_utterances.csv`.

## 3. Spelling normalization (`spl_correction/`)

This step standardizes Spanish words written with Guaraní spelling (for example *pórke* → *porque*).

- `build_spl_vocab_bank.py` asks gpt-oss-120b to find candidate respellings, 25 utterances at a
  time.
- `spl_vocab_bank.csv` is the final 160-entry bank after manual review.
- `apply_spl_corrections.py` and `add_final_corrected_column.py` apply the bank and write the
  corrected reference column, keeping the original text for audit.

Both LLM scripts call Indiana University's ReaLLMs service and read the API key from
`~/.reallms_secrets` (`REALLMS_API_KEY=...`).

## 4. Word-level language tagging (`lid_tagging/`)

- `run_lid_reverification.py` (with its `.sbatch` launcher) sends each utterance to gemma-4-31B-it,
  which tags every word as Spanish, Guaraní, *jehe'a*, or other.
- `fix_lid_fillers.py` then forces a fixed list of fillers and interjections to "other".

Each utterance's category comes from these tags.

## 5. Splits (`splits/`)

- `rebuild_test_training_v2.py` builds both the 150-utterance test set and the train/dev splits,
  despite the "v2" in its name.
- `make_moe_manifests.py` writes the shared JSONL test manifest used for scoring.
