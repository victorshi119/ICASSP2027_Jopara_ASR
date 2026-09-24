#!/bin/bash
# Pretrained OmniASR LLM 7B v2 baseline inference — no fine-tuning, no LoRA.
# Evaluates the pretrained model on all 4 v3 test subsets across three language modes:
#   grn_Latn  gug_Latn  spa_Latn
#
# All outputs go under this directory (results/June_09_baseline/).
# Logs go to results/June_09_baseline/logs/.
#
# Optional env var:
#   OVERRIDE_LANG   Run a single language mode instead of all three.
#                   e.g. OVERRIDE_LANG=grn_Latn sbatch run_baseline.sbatch
#
# Submit from this directory:
#   cd /N/project/icassp2026/Jopara_ASR_models_js2/results/June_09_baseline
#   sbatch run_baseline.sbatch

#SBATCH --job-name=baseline_v3
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/results/June_09_baseline/logs/baseline_v3.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/results/June_09_baseline/logs/baseline_v3.%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=122G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-interactive
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
OMNILINGUAL_ASR_ROOT="${OMNILINGUAL_ASR_ROOT:-${PROJECT_ROOT}/omnilingual-asr}"

FINETUNE_SCRIPTS_PRIMARY="${PROJECT_ROOT}/results/Guarani_only_result/finetune_llm_7b/scripts"
FINETUNE_SCRIPTS_FALLBACK="${PROJECT_ROOT}/results/Guarani_only_result/finetune_hdmole_7b/scripts"
if [[ -f "${FINETUNE_SCRIPTS_PRIMARY}/eval_debug_manifest.py" ]]; then
  FINETUNE_SCRIPTS="${FINETUNE_SCRIPTS_PRIMARY}"
else
  FINETUNE_SCRIPTS="${FINETUNE_SCRIPTS_FALLBACK}"
fi

LORA_7B_SPA="${JOPARA_ROOT}/lora_7b_spa"
TEST_SUBSETS_DIR="${PROJECT_ROOT}/updated_csvs/v2_test_subsets"
BASELINE_OUT="/N/project/icassp2026/Jopara_ASR_models_js2/results/June_09_baseline"
MANIFEST_DIR="${BASELINE_OUT}/manifests"

mkdir -p "${BASELINE_OUT}/logs" "${MANIFEST_DIR}"

log_out="${BASELINE_OUT}/logs/baseline_v3.${SLURM_JOB_ID}.out"
log_echo() { echo "$@"; }

write_exit_status() {
  local exit_code="${1:-$?}"
  echo "=== JOB EXIT STATUS ==="
  echo "Exit code: ${exit_code}"
  echo "Outputs: ${BASELINE_OUT}"
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

log_echo "===== Baseline v3 inference started at $(date -Iseconds 2>/dev/null || date) ====="
log_echo "TEST_SUBSETS_DIR=${TEST_SUBSETS_DIR}"
log_echo "BASELINE_OUT=${BASELINE_OUT}"

if [[ ! -d "${TEST_SUBSETS_DIR}" ]]; then
  log_echo "ERROR: TEST_SUBSETS_DIR not found: ${TEST_SUBSETS_DIR}"; exit 1
fi

# Language modes to evaluate — default all three; override with OVERRIDE_LANG env var
if [[ -n "${OVERRIDE_LANG:-}" ]]; then
  LANG_MODES=("${OVERRIDE_LANG}")
else
  LANG_MODES=("grn_Latn" "gug_Latn" "spa_Latn")
fi
log_echo "Language modes: ${LANG_MODES[*]}"

SUBSETS=("test_guarani_only" "test_spanish_only" "test_mixed" "test_jehea")

# --- [1/3] Build test manifests (once, shared across all lang modes) ---
log_echo ""
log_echo "===== [1/3] Build test manifests from v2_test_subsets ====="
"${PYTHON}" "${FINETUNE_SCRIPTS}/make_test_subsets_manifests.py" \
  --test-subsets-dir "${TEST_SUBSETS_DIR}" \
  --output-dir "${MANIFEST_DIR}" \
  --no-skip-missing >> "${log_out}" 2>&1
if [[ $? -ne 0 ]]; then log_echo "ERROR: make_test_subsets_manifests.py failed"; exit 1; fi

for split in "${SUBSETS[@]}"; do
  if [[ -f "${MANIFEST_DIR}/${split}.tsv" ]]; then
    n_utts=$(( $(wc -l < "${MANIFEST_DIR}/${split}.tsv") - 1 ))
    log_echo "  ${split}: ${n_utts} utterances"
  else
    log_echo "  WARNING: ${split}.tsv not found — will be skipped during inference"
  fi
done

# --- [2/3] Inference per language mode (no checkpoint = pretrained model only) ---
log_echo ""
log_echo "===== [2/3] Pretrained model inference (omniASR_LLM_7B_v2, no checkpoint, no LoRA) ====="

for lang in "${LANG_MODES[@]}"; do
  EVAL_OUT="${BASELINE_OUT}/${lang}"
  mkdir -p "${EVAL_OUT}"
  log_echo ""
  log_echo "--- Language mode: ${lang} -> ${EVAL_OUT} ---"

  for split in "${SUBSETS[@]}"; do
    if [[ ! -f "${MANIFEST_DIR}/${split}.tsv" ]]; then
      log_echo "  Skipping ${split} (no manifest)"
      continue
    fi
    n_utts=$(( $(wc -l < "${MANIFEST_DIR}/${split}.tsv") - 1 ))
    log_echo "  [${split}] ${n_utts} utterances, lang=${lang} ..."
    "${PYTHON}" "${FINETUNE_SCRIPTS}/eval_debug_manifest.py" \
      --manifest-dir "${MANIFEST_DIR}" \
      --output-csv "${EVAL_OUT}/${split}_eval.csv" \
      --split "${split}" \
      --model-card "omniASR_LLM_7B_v2" \
      --override-lang "${lang}" \
      --batch-size 2 >> "${log_out}" 2>&1
    if [[ $? -ne 0 ]]; then
      log_echo "  ERROR: inference failed for ${split} (${lang})"
      continue
    fi
    log_echo "  -> ${EVAL_OUT}/${split}_eval.csv"
  done
done

# --- [3/3] Build predictions and metrics per lang mode ---
log_echo ""
log_echo "===== [3/3] Build predictions and metrics ====="

for lang in "${LANG_MODES[@]}"; do
  EVAL_OUT="${BASELINE_OUT}/${lang}"
  PRED_DIR="${EVAL_OUT}/predictions"
  mkdir -p "${PRED_DIR}"
  log_echo ""
  log_echo "--- ${lang} predictions ---"

  for split in "${SUBSETS[@]}"; do
    eval_csv="${EVAL_OUT}/${split}_eval.csv"
    if [[ ! -f "${eval_csv}" ]]; then
      log_echo "  Skipping ${split} (no eval CSV)"
      continue
    fi
    "${PYTHON}" "${LORA_7B_SPA}/build_predictions_from_eval_csv.py" \
      --eval-csv "${eval_csv}" \
      --subset "${split}" \
      --output-dir "${PRED_DIR}" >> "${log_out}" 2>&1
    log_echo "  -> ${PRED_DIR}/${split}.ref.txt + .hyp.txt"
  done

  METRICS_JSON="${EVAL_OUT}/metrics.json"
  if [[ -f "${LORA_7B_SPA}/metrics.py" ]]; then
    for split in "${SUBSETS[@]}"; do
      ref="${PRED_DIR}/${split}.ref.txt"
      hyp="${PRED_DIR}/${split}.hyp.txt"
      if [[ -f "${ref}" && -f "${hyp}" ]]; then
        "${PYTHON}" "${LORA_7B_SPA}/metrics.py" \
          --ref "${ref}" \
          --hyp "${hyp}" \
          --subset "${split}" \
          --output "${METRICS_JSON}" >> "${log_out}" 2>&1
      fi
    done
    log_echo "  Metrics: ${METRICS_JSON}"
  fi
done

log_echo ""
log_echo "===== Done at $(date) ====="
log_echo "Results in ${BASELINE_OUT}/"
for lang in "${LANG_MODES[@]}"; do
  log_echo "  ${lang}/   <- eval CSVs, predictions/, metrics.json"
done
exit 0
