# Evaluation

## Inference (`inference/`)

Which script produced each row of Table 1:

| Row | System | Script |
|---|---|---|
| 1 | Whisper-large-v3, automatic language choice | `run_whisper_inference.py` |
| 2 | MMS-1B-all with its Guaraní adapter | `run_mms_inference.py` |
| 3 | OmniASR, auto-detect | `eval_debug_manifest.py` with an **empty** language column (`batches/run_row3_autodetect.sbatch`) |
| 4a–4c | OmniASR in the `grn_Latn` / `gug_Latn` / `spa_Latn` language mode | `eval_debug_manifest.py` (`batches/run_baseline_language_modes.sbatch`) |
| 5–7 | Single LoRA adapters | `eval_all_branches.py` |
| 8–9 | Mixture of experts | `training/moe_router/infer_jmole_asr.py` |

`make_test_subsets_manifests.py` builds the test manifests that `eval_debug_manifest.py` reads. In
that manifest's third column, an empty value means auto-detect; any language code **forces** that
language. An earlier version of the script always wrote `spa_Latn`, which silently turned the
"auto-detect" baseline into forced Spanish. The version here leaves the column empty by default.
Row 3's manifest is rebuilt with:

```bash
python evaluation/inference/make_test_subsets_manifests.py \
  --test-subsets-dir $JOPARA_ROOT/updated_csvs/v2_test_subsets --output-dir manifests \
  --combined-split v2_test_all --max-duration 0
```

## Scoring (`scoring/`)

All rows are scored the same way:

- Corpus-level CER and WER from SCTK `sclite`, via `metrics.py`.
- Whitespace is collapsed. Case and diacritics are kept.
- No other post-processing.
- The references are the 150 verified test utterances.

| Script | Purpose |
|---|---|
| `make_table1.py` | Table 1 as printed, plus per-utterance error counts (writes `results/table1/`) |
| `rescore_table1.py` | The original scorer. `make_table1.py` imports its loaders and normalization. |
| `row8_seeds.py`, `row9_seeds.py` | Per-seed scores for rows 8–9, and paired bootstrap CIs against rows 3 and 4a |
| `ci_check.py` | Checks the Results-section CIs. Runs a paired bootstrap that resamples utterances and, separately, whole recording sessions, and compares each adapter with its matched no-adapter mode. |
| `bootstrap_significance.py` | General paired bootstrap between any two systems |
| `postprocess_validation.py` | **Not part of the scoring.** Old test-derived normalization rules, kept only so `rescore_table1.py` can check it matches the earlier pipeline's numbers. |

`sclite` comes from SCTK; we used conda-forge `sctk`. Set it on `PATH` or edit `SCLITE` in
`metrics.py`.
