#!/bin/bash
# The three LoRA adapters in Table 1 (rows 5-7), exactly as submitted on IU BigRed200.
# Taken from the original submit_b2_ablations.sh / submit_b3_ablations.sh, keeping only these runs.
#
# Row 5  Spanish adapter          B2_spa_d010      rank 16, lr 1e-4, spa_Latn
# Row 6  Guarani adapter          B3_grn_cv_cegpa  rank 16, lr 2e-5, grn_Latn
#        train_cv_cegpa.csv = 2,569 Common Voice clips + 19 Guarani-only + 871 code-switching
#        CEGPA utterances (3,459 rows; job 7368272 log: "Wrote 3459 train, 286 dev")
# Row 7  Code-switching adapter   B2_cs_r32_spa    rank 32, lr 1e-4, spa_Latn
#
# SPLITS: the split CSVs listed in data/metadata/ (built from your own copy of CEGPA + Common Voice).
# The launchers use the original project layout; see training/README.md.
set -euo pipefail
PROJECT_ROOT="${JOPARA_ROOT:-/N/project/icassp2026/Jopara_ASR_models_js2}"
SPLITS="${PROJECT_ROOT}/updated_csvs/training_splits_v3"
CONFIGS="$(cd "$(dirname "$0")/../config" && pwd)"
cd "$(dirname "$0")"

LORA_RUN_ID=B2_spa_d010 TRAIN_CSV="${SPLITS}/train_spanish_only.csv" DEV_CSV="${SPLITS}/dev_spanish_only.csv" \
  OVERRIDE_LANG=spa_Latn LORA_CONFIG="${CONFIGS}/b2_spa_d010.yaml" MANIFEST_SUBDIR=manifests_b2_spa_d010 \
  sbatch run_lora_v3.sbatch

LORA_RUN_ID=B3_grn_cv_cegpa TRAIN_CSV="${SPLITS}/train_cv_cegpa.csv" DEV_CSV="${SPLITS}/dev_guarani_only.csv" \
  OVERRIDE_LANG=grn_Latn LORA_CONFIG="${CONFIGS}/b3_grn_base.yaml" \
  sbatch run_lora_b3.sbatch

LORA_RUN_ID=B2_cs_r32_spa TRAIN_CSV="${SPLITS}/train_cs_jopara.csv" DEV_CSV="${SPLITS}/dev_cs_jopara.csv" \
  OVERRIDE_LANG=spa_Latn LORA_CONFIG="${CONFIGS}/b2_cs_r32.yaml" MANIFEST_SUBDIR=manifests_b2_cs_r32_spa \
  sbatch run_lora_v3.sbatch

# ---- Experts used only inside the mixture of experts (rows 8-9) ----
# Row 8's code-switching expert: the same rank-32 adapter, trained in the Guarani language mode.
LORA_RUN_ID=B2_cs_r32_grn TRAIN_CSV="${SPLITS}/train_cs_jopara.csv" DEV_CSV="${SPLITS}/dev_cs_jopara.csv" \
  OVERRIDE_LANG=grn_Latn LORA_CONFIG="${CONFIGS}/b2_cs_r32.yaml" MANIFEST_SUBDIR=manifests_b2_cs_r32_grn \
  sbatch run_lora_v3.sbatch

# Rows 8-9's Guarani expert: the row-6 recipe retrained once per seed (jobs 8193726-8, 2026-09-10).
# Same 3,459-row list as row 6 with repaired audio paths; 3,445 rows resolved to audio at manifest time.
for SEED in 13 42 123; do
  SEED=${SEED} LORA_RUN_ID=B3_grn_cv_cegpa_seed${SEED} TRAIN_CSV="${SPLITS}/train_cv_cegpa.csv" \
    DEV_CSV="${SPLITS}/dev_guarani_only.csv" OVERRIDE_LANG=grn_Latn LORA_CONFIG="${CONFIGS}/b3_grn_base.yaml" \
    MANIFEST_SUBDIR=manifests_b3_seed${SEED} sbatch run_lora_b3.sbatch
done
