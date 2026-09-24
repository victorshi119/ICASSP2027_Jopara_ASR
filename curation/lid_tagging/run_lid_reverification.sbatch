#!/bin/bash
# Task 2: Word-level LID re-verification via IU ReaLLMs (gemma-4-31B-it).
#
# Uses final_corrected_reference_norm from Everything.csv as input.
# Compute is on ReaLLMs servers (HTTP API) — no GPU needed on this node.
# Resume-safe: re-submit this job if interrupted; it continues from the last cached batch.
#
# Submit:
#   sbatch QC_plans/run_lid_reverification.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /N/project/icassp2026/Jopara_ASR_models/logs/lid_reverification.$SLURM_JOB_ID.out
#
#SBATCH --job-name=lid_reverify
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models/logs/lid_reverification.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models/logs/lid_reverification.%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH -A r01360

set -uo pipefail
set -x

export PYTHONUNBUFFERED=1
# Keep real HOME so ~/.reallms_secrets resolves correctly
REAL_HOME="${HOME}"

WORKDIR=/N/project/icassp2026/Jopara_ASR_models
PYTHON=/N/project/icassp2026/humdial/soEmo2/conda_envs/OSUM-EChat/bin/python3
SCRIPT=${WORKDIR}/QC_plans/run_lid_reverification.py

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working dir: ${WORKDIR}"

cd "${WORKDIR}"

# Load API key from real home directory
SECRETS_FILE="${REAL_HOME}/.reallms_secrets"
if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "ERROR: ${SECRETS_FILE} not found"
  exit 1
fi
source "${SECRETS_FILE}" 2>/dev/null || true
export REALLMS_API_KEY

${PYTHON} "${SCRIPT}"

echo "Job finished: $(date)"
