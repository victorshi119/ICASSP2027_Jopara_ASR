#!/usr/bin/env python3
"""
0920: rescore the kept Table 1 rows WITHOUT the ChatGPT-derived post-processing rules.

Why: postprocess_norm.py's rules were derived from the zero-shot model's errors on the 150-utterance TEST set
(results/June_21_Sclite_Confusion_Pair_PostProcessing/), so any number scored with them is test-informed.
"Raw" here = sclite corpus-level CER/WER on the stored hypotheses with whitespace collapse only
(metrics.normalize), against the same references.

Validation: each system is ALSO scored with the old pipeline (postprocess_norm.postprocess applied to ref and hyp)
and compared with the values printed in the current Table 1. If those match, the hypothesis files are the right ones.

Rows covered: 1-8 and 11-12 (rows 9, 10, 13, 14 are dropped from the paper).
Run (light, sclite only):  python3 rescore_table1.py
"""
from __future__ import annotations
import os
import csv, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
OUT = Path(os.environ.get("JOPARA_OUT", Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_metrics, normalize          # noqa: E402
from postprocess_validation import postprocess                 # noqa: E402

MANIFEST = ROOT / "moe_lora/manifests/v2_test_moe_manifest.jsonl"
SUBSETS = ["test_guarani_only", "test_mixed", "test_jehea", "test_spanish_only"]
COLS = SUBSETS + ["overall"]

# Values currently printed in Table 1 (old pipeline): CER / WER per column.
PAPER = {
    1: [(89.0, 128.2), (45.6, 62.4), (55.9, 79.7), (20.3, 42.3), (47.4, 66.7)],
    2: [(30.0, 82.2), (34.1, 78.6), (30.8, 76.8), (26.4, 64.8), (31.6, 75.7)],
    3: [(40.1, 89.9), (29.5, 49.1), (29.0, 57.3), (13.6, 24.9), (27.5, 48.8)],
    4: [(33.3, 76.7), (25.6, 45.7), (24.7, 53.4), (12.6, 24.7), (23.7, 45.4)],
    5: [(54.8, 119.5), (39.2, 59.5), (45.0, 83.2), (12.2, 24.1), (37.4, 62.0)],
    6: [(30.9, 83.6), (28.3, 49.7), (36.5, 69.4), (14.2, 27.9), (28.3, 52.3)],
    7: [(32.6, 78.0), (32.3, 51.5), (34.7, 67.8), (11.5, 25.3), (29.4, 52.0)],
    8: [(31.5, 84.9), (29.6, 46.6), (33.4, 66.1), (11.5, 22.4), (27.7, 48.9)],
    11: [(32.7, 80.4), (32.0, 51.8), (34.9, 68.8), (12.4, 24.8), (29.5, 52.5)],
    12: [(29.9, 72.4), (22.2, 37.3), (25.2, 47.2), (10.5, 20.5), (21.6, 38.6)],
}
NAMES = {1: "Whisper-large-v3, automatic language choice", 2: "MMS-1B-all, Guaraní adapter",
         3: "OmniASR, unconditioned (baseline)", 4: "OmniASR, fixed Guaraní–Latin conditioning",
         5: "Spanish adapter", 6: "Guaraní adapter", 7: "Code-switching adapter",
         8: "Soft router over all three adapters", 11: "Route by true language category (†)",
         12: "Route to the best adapter for each utterance (†)"}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


manifest = jl(MANIFEST)
ids = [r["utt_id"] for r in manifest]
assert len(ids) == 150
subset_of = {r["utt_id"]: r["subset"] for r in manifest}
ref_of = {r["utt_id"]: r["reference"] for r in manifest}
cat_of = {r["utt_id"]: r.get("v2_category", "") for r in manifest}


def stem(p):
    return Path(p).stem


def from_eval_csvs(folder):
    out = {}
    for s in SUBSETS:
        with open(Path(folder) / f"{s}_eval.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[stem(r["audio_path"])] = r["hypothesis"]
    return out


def from_branch(name):
    return {r["utt_id"]: r["hypothesis"] for r in jl(ROOT / f"moe_lora/predictions/all_branches/{name}.jsonl")}


systems = {}
systems[1] = from_eval_csvs(ROOT / "results/June_09_baseline/whisper_large/auto")
systems[2] = from_eval_csvs(ROOT / "results/June_09_baseline/mms/grn")
with open(ROOT / "results/rebuttal_stats/row3_baseline/v2_test_all_eval.csv", newline="", encoding="utf-8") as f:
    systems[3] = {stem(r["audio_path"]): r["hypothesis"] for r in csv.DictReader(f)}
systems[4] = from_branch("Z0_base_grn")
systems[5] = from_branch("E0_spa_lora")
systems[6] = from_branch("E1_grn_lora_cegpa")
systems[7] = from_branch("E3_cs_spa_lora")
systems[8] = {r["utt_id"]: r["hyp_jmole"] for r in jl(ROOT / "results/June_15_Multitask_Audio_LLM/jopara_mole_asr/hypothesis_packs/jmole_grn_test_hypotheses.jsonl")}

# oracles, reconstructed exactly as in rescore_all.py
branch_rows = {p.stem: {r["utt_id"]: r for r in jl(p)} for p in (ROOT / "moe_lora/predictions/all_branches").glob("*.jsonl")}
R0_map = {"guarani_only": "E3_cs_spa_lora", "jehe'a": "E3_cs_spa_lora", "mixed": "E5_grn_csinit_lora", "spanish_only": "E0_spa_lora"}
systems[11] = {u: branch_rows[R0_map.get(cat_of[u], "E3_cs_spa_lora")][u]["hypothesis"] for u in ids}
systems[12] = {}
for u in ids:
    best = min(branch_rows, key=lambda b: branch_rows[b].get(u, {}).get("wer", 1e9))
    systems[12][u] = branch_rows[best][u]["hypothesis"]


def score(refs, hyps):
    m = compute_metrics(refs, hyps)
    return round(m["cer_corpus"] * 100, 1), round(m["wer_corpus"] * 100, 1)


def score_system(hyp_of, post):
    prep = (lambda t: postprocess(normalize(t))) if post else normalize
    by = defaultdict(lambda: ([], []))
    allr, allh = [], []
    for u in ids:
        if u not in hyp_of:
            raise KeyError(f"missing utterance {u}")
        r, h = prep(ref_of[u]), prep(hyp_of[u])
        by[subset_of[u]][0].append(r); by[subset_of[u]][1].append(h)
        allr.append(r); allh.append(h)
    res = [score(*by[s]) for s in SUBSETS]
    res.append(score(allr, allh))
    return res


results, report = {}, []
for row in sorted(systems):
    miss = [u for u in ids if u not in systems[row]]
    old = score_system(systems[row], post=True)
    raw = score_system(systems[row], post=False)
    paper = PAPER[row]
    dev = max(abs(o[0] - p[0]) for o, p in zip(old, paper))
    results[row] = {"name": NAMES[row], "raw": raw, "old_pipeline": old, "paper_table": paper,
                    "max_abs_cer_diff_old_vs_paper": round(dev, 2), "missing": len(miss)}
    report.append((row, dev))
    print(f"row {row:2d} {NAMES[row][:44]:44s} old-vs-paper max |dCER|={dev:4.2f}  raw overall CER/WER={raw[4]}")

(OUT / "table1_raw.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

lines = ["# Table 1 rescored WITHOUT the test-derived post-processing rules (CER / WER, %)",
         "", "Columns: Guaraní-only | inter-sentential (mixed) | jehe'a | Spanish-only | overall. Corpus-level sclite.",
         "'old' = the values currently in the paper (validated by re-running the old pipeline). 'raw' = new.", ""]
lines.append("| Row | System | " + " | ".join(["Guaraní", "Inter-sent.", "Jehe'a", "Spanish", "Overall"]) + " | old-vs-paper |")
lines.append("|---|---|---|---|---|---|---|---|")
for row, r in results.items():
    cells = " | ".join(f"{c:.1f} / {w:.1f}" for c, w in r["raw"])
    lines.append(f"| {row} | {r['name']} | {cells} | {r['max_abs_cer_diff_old_vs_paper']} |")
(OUT / "table1_raw.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT / "table1_raw.md")
