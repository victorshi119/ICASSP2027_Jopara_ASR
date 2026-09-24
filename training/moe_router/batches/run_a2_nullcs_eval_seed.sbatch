#!/bin/bash
# A2-NullCS eval for one multi-seed run (rebuttal #6, 2026-08-11).
# Same pipeline as results/June_24_Experiments/run_a2_nullcs_eval.sbatch, but router+E_grn
# come from a specific seed (E_spa/E_cs fixed at production values). Requires SEED env var.
#
# Submit (after the matching router + B3 seed jobs complete):
#   SEED=13 sbatch results/rebuttal_stats/multiseed/run_a2_nullcs_eval_seed.sbatch

#SBATCH --job-name=a2_nullcs_seed
#SBATCH --output=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/a2_nullcs_seed.%j.out
#SBATCH --error=/N/project/icassp2026/Jopara_ASR_models_js2/results/rebuttal_stats/multiseed/logs/a2_nullcs_seed.%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=122G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -A r01360
# NOTE: gpu-interactive's QOS caps MaxSubmitPU=1 (can't queue 3 concurrently as
# dependency jobs), so this uses the gpu partition instead.

set -uo pipefail
set -x
export PYTHONUNBUFFERED=1

SEED="${SEED:?ERROR: SEED must be set (13|42|123)}"

JS2_ROOT="/N/project/icassp2026/Jopara_ASR_models_js2"
OMNI_ROOT="${JS2_ROOT}/omnilingual-asr"
J15="${JS2_ROOT}/results/June_15_Multitask_Audio_LLM"

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

source /N/slate/wencshi/miniforge3/etc/profile.d/conda.sh
conda activate omniasr_fairseq2
export PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/bin:/N/slate/wencshi/miniforge3/condabin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/N/slate/wencshi/conda_envs/omniasr_fairseq2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${OMNI_ROOT}:${OMNI_ROOT}/src:${JS2_ROOT}:${JS2_ROOT}/results/June_21_Sclite_Confusion_Pair_PostProcessing:${PYTHONPATH:-}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

CONFIG="${JS2_ROOT}/results/rebuttal_stats/configs/a2_nullcs_router_seed${SEED}.yaml"
ROUTER_CKPT="/N/scratch/wencshi/icassp2026_rebuttal/router_seeds/seed_${SEED}/router_best.pt"
OUTDIR="${JS2_ROOT}/results/rebuttal_stats/multiseed/a2_nullcs_seed${SEED}"
HYP_FILE="${J15}/jopara_mole_asr/hypothesis_packs/a2_nullcs_seed${SEED}_a2_test_hypotheses.jsonl"
mkdir -p "${OUTDIR}"

echo "===== A2-NullCS seed=${SEED} eval started at $(date) ====="
for f in "${CONFIG}" "${ROUTER_CKPT}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}"; exit 1; }
done

cd "${JS2_ROOT}"
python "${J15}/jopara_mole_asr/scripts/infer_jmole_asr.py" \
    --config      "${CONFIG}" \
    --split       test \
    --router-ckpt "${ROUTER_CKPT}" \
    --batch-size  1

echo ""
echo "===== Inference complete, scoring... ====="

python - << PYEOF
import json
import sys
from pathlib import Path

JS2 = Path("/N/project/icassp2026/Jopara_ASR_models_js2")
sys.path.insert(0, str(JS2))
sys.path.insert(0, str(JS2 / "results/June_21_Sclite_Confusion_Pair_PostProcessing"))

from metrics import compute_metrics, normalize
from postprocess_norm import postprocess

SEED = "${SEED}"
HYP_FILE = JS2 / f"results/June_15_Multitask_Audio_LLM/jopara_mole_asr/hypothesis_packs/a2_nullcs_seed{SEED}_a2_test_hypotheses.jsonl"
SUMMARY  = JS2 / f"results/rebuttal_stats/multiseed/a2_nullcs_seed{SEED}/summary.json"

SUBSETS = {"guarani_only": "Guaraní-only", "spanish_only": "Spanish-only", "mixed": "Mixed", "jehe'a": "Jehe'a"}
SUBSET_ALIAS = {"jehea": "jehe'a", "jehe'a": "jehe'a"}

rows = [json.loads(l) for l in HYP_FILE.read_text().splitlines() if l.strip()]
manifest = [json.loads(l) for l in (JS2 / "moe_lora/manifests/v2_test_moe_manifest.jsonl").read_text().splitlines() if l.strip()]
utt2cat = {r["utt_id"]: r.get("v2_category","") for r in manifest}

def norm(t): return normalize(postprocess(str(t or "")))

by_subset = {k: ([], []) for k in SUBSETS}
all_refs, all_hyps = [], []
for r in rows:
    ss = utt2cat.get(r["utt_id"], SUBSET_ALIAS.get(r.get("regime",""), r.get("regime","")))
    ref = norm(r["reference"])
    hyp = norm(r.get("hyp_jmole", ""))
    if ss in by_subset:
        by_subset[ss][0].append(ref)
        by_subset[ss][1].append(hyp)
    all_refs.append(ref)
    all_hyps.append(hyp)

result = {"seed": int(SEED)}
overall_m = compute_metrics(all_refs, all_hyps)
result["overall"] = {"n": len(all_refs), "cer": overall_m["cer_corpus"]*100, "wer": overall_m["wer_corpus"]*100}
for ss, label in SUBSETS.items():
    refs, hyps = by_subset[ss]
    if refs:
        m = compute_metrics(refs, hyps)
        result[ss] = {"n": len(refs), "cer": m["cer_corpus"]*100, "wer": m["wer_corpus"]*100}

print(json.dumps(result, indent=2))
SUMMARY.write_text(json.dumps(result, indent=2))
print(f"\nSummary -> {SUMMARY}")
PYEOF

echo ""
echo "===== A2-NullCS seed=${SEED} eval complete at $(date) ====="
