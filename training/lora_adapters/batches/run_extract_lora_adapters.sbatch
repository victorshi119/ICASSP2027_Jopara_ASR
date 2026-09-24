#!/bin/bash
# Extract lora_adapters.pt for one or more multi-seed LoRA run dirs, using the
# fixed standalone extractor (see extract_lora_adapter.py for why train_lora.py's
# built-in extraction silently fails on some seeds).
#
# Submit:
#   RUN_DIRS="/N/scratch/.../B4_spa_grn_seed13 /N/scratch/.../B4_spa_grn_seed42" \
#     sbatch results/rebuttal_stats/multiseed/run_extract_lora_adapters.sbatch

#SBATCH --job-name=extract_lora
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/extract_lora.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/extract_lora.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH -A r01360

set -uo pipefail
set -x
export PYTHONUNBUFFERED=1

JS2_ROOT="/N/project/icassp2026/Jopara_ASR_models_js2"
SCRIPT="${JS2_ROOT}/results/rebuttal_stats/multiseed/scripts/extract_lora_adapter.py"

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

RUN_DIRS="${RUN_DIRS:?ERROR: RUN_DIRS must be set (space-separated list of run output dirs)}"

echo "===== extract_lora_adapters started at $(date) ====="
overall_ec=0
for d in ${RUN_DIRS}; do
  echo ""
  echo "----- ${d} -----"
  python "${SCRIPT}" --run-dir "${d}"
  ec=$?
  echo "exit=${ec}"
  [[ ${ec} -ne 0 ]] && overall_ec=1
done
echo ""
echo "===== extract_lora_adapters done at $(date), overall_ec=${overall_ec} ====="
exit ${overall_ec}
