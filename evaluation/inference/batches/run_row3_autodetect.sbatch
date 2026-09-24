#!/bin/bash
# Row-3 baseline (OmniASR, unconditioned, automatic language choice) re-run
# against the CURRENT v2 150-utterance test manifest (moe_lora/manifests/
# v2_test_moe_manifest.jsonl), so it can be paired utterance-by-utterance
# with the A2-NullCS (row 9) hypothesis packs for a valid bootstrap CI.
#
# The original row-3 numbers (results/June_09_baseline/auto_lang/) were
# scored against an OLDER, differently-composed test set (pre-rebalancing:
# 33 Spanish-only/25 Jehe'a instead of today's 25/33), so those predictions
# cannot be paired against the current row 9 packs utterance-for-utterance.
# This re-run uses the identical inference call (no --override-lang,
# no --fallback-lang => pipeline auto-detects language per utterance) on the
# exact 150 utterances Table 1 actually reports.
#
# Submit:
#   cd /N/project/icassp2026/Jopara_ASR_models_js2
#   sbatch results/rebuttal_stats/row3_baseline/run_row3_baseline_v2.sbatch

#SBATCH --job-name=row3_baseline_v2
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/row3_baseline/logs/row3.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/row3_baseline/logs/row3.%j.err
#SBATCH --time=2:00:00
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

OMNILINGUAL_ASR_ROOT="${PROJECT_ROOT}/omnilingual-asr"
FINETUNE_SCRIPTS="${PROJECT_ROOT}/results/Guarani_only_result/finetune_llm_7b/scripts"
MANIFEST_DIR="${PROJECT_ROOT}/results/rebuttal_stats/row3_baseline/manifests"
OUT_DIR="${PROJECT_ROOT}/results/rebuttal_stats/row3_baseline"
EVAL_CSV="${OUT_DIR}/v2_test_all_eval.csv"

mkdir -p "${OUT_DIR}/logs"

source /N/slate/wencshi/miniforge3/etc/profile.d/conda.sh
conda activate omniasr_fairseq2
export PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/bin:/N/slate/wencshi/miniforge3/condabin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${OMNILINGUAL_ASR_ROOT}:${OMNILINGUAL_ASR_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

echo "===== Row-3 baseline (v2 test set) started at $(date) ====="
echo "No --override-lang, no --fallback-lang => pipeline auto-detects language"

python "${FINETUNE_SCRIPTS}/eval_debug_manifest.py" \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-csv "${EVAL_CSV}" \
    --split "v2_test_all" \
    --model-card "omniASR_LLM_7B_v2" \
    --batch-size 2

echo ""
echo "===== Row-3 baseline done at $(date) ====="
echo "Eval CSV: ${EVAL_CSV}"
