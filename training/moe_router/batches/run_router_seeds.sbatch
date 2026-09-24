#!/bin/bash
# Multi-seed A2 router (SoftRegimeRouter) retrain — rebuttal #6 (2026-08-11).
# Runs seeds 13/42/123 sequentially. CPU-only (pre-extracted embeddings, fast path),
# so this uses the `general` partition, not gpu/gpu-interactive.
#
# Submit:
#   sbatch results/rebuttal_stats/multiseed/run_router_seeds.sbatch

#SBATCH --job-name=router_seeds
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/router_seeds.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/router_seeds.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH -A r01360

set -uo pipefail
set -x
export PYTHONUNBUFFERED=1

JS2_ROOT="/N/project/icassp2026/Jopara_ASR_models_js2"
J15="${JS2_ROOT}/results/June_15_Multitask_Audio_LLM"
CONFIG="${JS2_ROOT}/results/rebuttal_stats/configs/jmole_grn_bigred200.yaml"
OUT_ROOT="/N/scratch/wencshi/icassp2026_rebuttal/router_seeds"

export HOME="/N/slate/wencshi"
export FAIRSEQ2_CACHE_DIR="/N/slate/wencshi/.cache/fairseq2/assets"
export XDG_CACHE_HOME="/N/slate/wencshi/.cache"
export XDG_CONFIG_HOME="/N/slate/wencshi/.config"
export XDG_DATA_HOME="/N/slate/wencshi/.local/share"
export PYTHONUSERBASE="/N/slate/wencshi/.local"
export TMPDIR="/N/slate/wencshi/.tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}"

source /N/slate/wencshi/miniforge3/etc/profile.d/conda.sh
conda activate omniasr_fairseq2
export PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/bin:/N/slate/wencshi/miniforge3/condabin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${JS2_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

for SEED in 13 42 123; do
  echo "===== Router seed=${SEED} started at $(date) ====="
  CKPT_DIR="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${CKPT_DIR}"
  python "${J15}/jopara_mole_asr/scripts/train_router_on_embeddings.py" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --checkpoint-dir "${CKPT_DIR}"
  ec=$?
  echo "===== Router seed=${SEED} exit=${ec} at $(date) ====="
  if [[ ${ec} -ne 0 ]]; then
    echo "ERROR: router seed ${SEED} failed"; exit ${ec}
  fi
done

echo "===== All router seeds complete at $(date) ====="
exit 0
