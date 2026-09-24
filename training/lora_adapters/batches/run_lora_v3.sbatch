#!/bin/bash
# v3 LoRA training — parameterized for all B1 runs (Spanish-only, Guarani-only, CS-Jopara).
# Pre-split CSVs from training_splits_v3/ are used directly; Step 1 (data prep) is skipped.
# Dataset card is named jopara_v3_${LORA_RUN_ID}_manifest (unique per run — no collision risk).
#
# Required env vars (set before sbatch or via submit_v3_runs.sh):
#   LORA_RUN_ID      Run identifier, e.g. B1_spa / B1_grn / B1_cs_gug
#   TRAIN_CSV        Absolute path to train_*.csv in training_splits_v3/
#   DEV_CSV          Absolute path to dev_*.csv in training_splits_v3/
#   OVERRIDE_LANG    Language mode: spa_Latn | grn_Latn | gug_Latn
#   LORA_CONFIG      Absolute path to config yaml
#   MANIFEST_SUBDIR  Unique subdir name, e.g. manifests_b1_spa
#
# Resume after OOM/timeout:
#   LORA_RUN_ID=B1_spa LORA_CONFIG=.../a0_baseline.yaml sbatch run_lora_v3_resume.sbatch

#SBATCH --job-name=lora_v3
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/lora_7b_spa/logs/out/lora_v3.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/lora_7b_spa/logs/err/lora_v3.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -A r01360

set -uo pipefail
set -x

export PYTHONUNBUFFERED=1
PROJECT_ROOT="/N/project/icassp2026/Jopara_ASR_models_js2"
export HOME="/N/slate/wencshi"
export FAIRSEQ2_CACHE_DIR="/N/slate/wencshi/.cache/fairseq2/assets"
export XDG_CACHE_HOME="/N/slate/wencshi/.cache"
export XDG_CONFIG_HOME="/N/slate/wencshi/.config"
export XDG_DATA_HOME="/N/slate/wencshi/.local/share"
export PYTHONUSERBASE="/N/slate/wencshi/.local"
export TMPDIR="/N/slate/wencshi/.tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_RUNTIME_DIR="${TMPDIR}/xdg_runtime"
mkdir -p "${TMPDIR}" "${XDG_RUNTIME_DIR}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}"

JOPARA_ROOT="${JOPARA_ROOT:-${PROJECT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/v3_lora_runs}"
OMNILINGUAL_ASR_ROOT="${OMNILINGUAL_ASR_ROOT:-${PROJECT_ROOT}/omnilingual-asr}"

FINETUNE_SCRIPTS_PRIMARY="${PROJECT_ROOT}/results/Guarani_only_result/finetune_llm_7b/scripts"
FINETUNE_SCRIPTS_FALLBACK="${PROJECT_ROOT}/results/Guarani_only_result/finetune_hdmole_7b/scripts"
if [[ -f "${FINETUNE_SCRIPTS_PRIMARY}/make_full_7b_manifest.py" ]]; then
  FINETUNE_SCRIPTS="${FINETUNE_SCRIPTS_PRIMARY}"
else
  FINETUNE_SCRIPTS="${FINETUNE_SCRIPTS_FALLBACK}"
fi
if [[ -f "${FINETUNE_SCRIPTS}/download_tokenizer.py" ]]; then
  TOKENIZER_SCRIPT="${FINETUNE_SCRIPTS}/download_tokenizer.py"
elif [[ -f "${FINETUNE_SCRIPTS_FALLBACK}/download_tokenizer.py" ]]; then
  TOKENIZER_SCRIPT="${FINETUNE_SCRIPTS_FALLBACK}/download_tokenizer.py"
else
  echo "ERROR: download_tokenizer.py not found"; exit 1
fi

LORA_RUN_ID="${LORA_RUN_ID:?ERROR: LORA_RUN_ID must be set (e.g. B1_spa)}"
TRAIN_CSV="${TRAIN_CSV:?ERROR: TRAIN_CSV must be set}"
DEV_CSV="${DEV_CSV:?ERROR: DEV_CSV must be set}"
OVERRIDE_LANG="${OVERRIDE_LANG:?ERROR: OVERRIDE_LANG must be set (spa_Latn|grn_Latn|gug_Latn)}"
LORA_CONFIG="${LORA_CONFIG:?ERROR: LORA_CONFIG must be set}"
MANIFEST_SUBDIR="${MANIFEST_SUBDIR:?ERROR: MANIFEST_SUBDIR must be set}"

# Unique dataset card name per run — avoids collision when runs execute simultaneously
DATASET_NAME="jopara_v3_${LORA_RUN_ID}_manifest"

MANIFEST_DIR="${RUN_ROOT}/${MANIFEST_SUBDIR}"
LORA_OUT_DIR="${RUN_ROOT}/lora_runs/${LORA_RUN_ID}"

LOG_DIR="${JOPARA_ROOT}/lora_7b_spa/logs"
mkdir -p "${LOG_DIR}/out" "${LOG_DIR}/err" "${MANIFEST_DIR}" "${LORA_OUT_DIR%/*}"

log_out="${LOG_DIR}/out/lora_v3.${SLURM_JOB_ID}.out"
log_echo() { echo "$@"; }  # stdout is captured by SLURM to --output file; no tee needed

write_exit_status() {
  local exit_code="${1:-$?}"
  echo "=== JOB EXIT STATUS ==="
  echo "Exit code: ${exit_code}"
  echo "Run: ${LORA_RUN_ID} | Lang: ${OVERRIDE_LANG} | Config: $(basename ${LORA_CONFIG})"
  echo "Outputs: ${LORA_OUT_DIR}"
  echo "Resume: LORA_RUN_ID=${LORA_RUN_ID} LORA_CONFIG=${LORA_CONFIG} sbatch run_lora_v3_resume.sbatch"
}
trap 'ec=$?; write_exit_status $ec; exit $ec' EXIT

source /N/slate/wencshi/miniforge3/etc/profile.d/conda.sh
conda activate omniasr_fairseq2
export PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/bin:/N/slate/wencshi/miniforge3/condabin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${OMNILINGUAL_ASR_ROOT}:${OMNILINGUAL_ASR_ROOT}/src:${JOPARA_ROOT}:${PYTHONPATH:-}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
PYTHON="python"

log_echo "===== LoRA v3 [${LORA_RUN_ID}] (${OVERRIDE_LANG}) started at $(date -Iseconds 2>/dev/null || date) ====="
log_echo "TRAIN_CSV=${TRAIN_CSV}"
log_echo "DEV_CSV=${DEV_CSV}"
log_echo "LORA_CONFIG=${LORA_CONFIG}"
log_echo "MANIFEST_DIR=${MANIFEST_DIR}"
log_echo "LORA_OUT_DIR=${LORA_OUT_DIR}"
log_echo "DATASET_NAME=${DATASET_NAME}"

if [[ ! -f "${TRAIN_CSV}" ]]; then log_echo "ERROR: TRAIN_CSV not found: ${TRAIN_CSV}"; exit 1; fi
if [[ ! -f "${DEV_CSV}" ]];   then log_echo "ERROR: DEV_CSV not found: ${DEV_CSV}";   exit 1; fi
if [[ ! -f "${LORA_CONFIG}" ]]; then log_echo "ERROR: LORA_CONFIG not found: ${LORA_CONFIG}"; exit 1; fi

# --- [1/4] Manifests ---
log_echo ""
log_echo "===== [1/4] Build manifests (lang=${OVERRIDE_LANG}, subdir=${MANIFEST_SUBDIR}) ====="
"${PYTHON}" "${FINETUNE_SCRIPTS}/make_full_7b_manifest.py" \
  --train-csv "${TRAIN_CSV}" \
  --dev-csv "${DEV_CSV}" \
  --output-dir "${RUN_ROOT}" \
  --override-lang "${OVERRIDE_LANG}" \
  --manifest-subdir "${MANIFEST_SUBDIR}" >> "${log_out}" 2>&1
if [[ $? -ne 0 ]]; then log_echo "ERROR: make_full_7b_manifest.py failed"; exit 1; fi

# --- [2/4] Tokenizer ---
log_echo ""
log_echo "===== [2/4] Pre-download tokenizer ====="
"${PYTHON}" "${TOKENIZER_SCRIPT}" --name "omniASR_tokenizer_written_v2" >> "${log_out}" 2>&1 || true

# --- [3/4] Dataset card (unique name = no collision with parallel runs) ---
CARDS_DIR="${OMNILINGUAL_ASR_ROOT}/src/omnilingual_asr/cards/datasets"
CARD_PATH="${CARDS_DIR}/${DATASET_NAME}.yaml"
mkdir -p "${CARDS_DIR}"
log_echo ""
log_echo "===== [3/4] Writing dataset card ${CARD_PATH} (data=${MANIFEST_DIR}) ====="
cat > "${CARD_PATH}" << EOF
name: ${DATASET_NAME}
dataset_family: manifest_asr_dataset
dataset_config:
  data: ${MANIFEST_DIR}
tokenizer_ref: omniASR_tokenizer_written_v2
EOF
if [[ ! -s "${CARD_PATH}" ]]; then log_echo "ERROR: Dataset card write failed"; exit 1; fi

# --- [4/4] LoRA training ---
log_echo ""
log_echo "===== [4/4] LoRA training (${LORA_RUN_ID}, ${OVERRIDE_LANG}, config=$(basename ${LORA_CONFIG})) ====="
(cd "${OMNILINGUAL_ASR_ROOT}" && "${PYTHON}" -m lora_7b_spa.train_lora \
  --config "${LORA_CONFIG}" \
  --output-dir "${LORA_OUT_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --model-card "omniASR_LLM_7B_v2") >> "${log_out}" 2>&1
exit_ok=$?
if [[ ${exit_ok} -ne 0 ]]; then
  log_echo "LoRA training exited with ${exit_ok} (OOM or timeout — resubmit run_lora_v3_resume.sbatch)"
  exit ${exit_ok}
fi
log_echo "LoRA adapters in ${LORA_OUT_DIR}"

log_echo "===== Done at $(date) ====="
exit 0
