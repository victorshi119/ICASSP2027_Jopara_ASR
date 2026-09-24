#!/usr/bin/env python3
"""Table 1 row 8 (mixture of experts over all three real adapters): per-seed RAW scores
(unrounded), mean +/- sample std over seeds 13/42/123, and per-seed paired bootstrap against
row 3 (auto-detect) and row 4a (Guarani mode). Mirrors results/0920_rescore_no_postnorm/
nullcs_seeds.py exactly (same normalize(), same subset grouping, same bootstrap), so the
row 8 and row 9 numbers are directly comparable cell-for-cell."""
import os
import json, random, re, statistics as st
from pathlib import Path
ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
src = open(Path(__file__).resolve().parent / "rescore_table1.py", encoding="utf-8").read().split("results, report = {}, []")[0]
ns = {"__name__": "rs", "__file__": str(Path(__file__).resolve().parent / "rescore_table1.py")}; exec(compile(src, "rs", "exec"), ns)
ids, ref_of, systems, normalize = ns["ids"], ns["ref_of"], ns["systems"], ns["normalize"]
compute_metrics, SUBSETS, subset_of = ns["compute_metrics"], ns["SUBSETS"], ns["subset_of"]
HP = ROOT / "results/June_15_Multitask_Audio_LLM/jopara_mole_asr/hypothesis_packs"
seeds = {13: "row8_seed13_a2_test_hypotheses.jsonl", 42: "row8_seed42_a2_test_hypotheses.jsonl", 123: "row8_seed123_a2_test_hypotheses.jsonl"}
def cols(hy):
    by = {s: ([], []) for s in SUBSETS}; ar, ah = [], []
    for u in ids:
        r, h = normalize(ref_of[u]), normalize(hy[u]); by[subset_of[u]][0].append(r); by[subset_of[u]][1].append(h); ar.append(r); ah.append(h)
    res = []
    for s in SUBSETS + ["all"]:
        r, h = (ar, ah) if s == "all" else by[s]; m = compute_metrics(r, h); res.append((m["cer_corpus"] * 100, m["wer_corpus"] * 100))
    return res
per = {}
for sd, f in seeds.items():
    hy = {json.loads(l)["utt_id"]: json.loads(l)["hyp_jmole"] for l in (HP / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    missing = [u for u in ids if u not in hy]
    assert not missing, f"seed {sd} missing {len(missing)} utterances"
    systems[sd] = hy; per[sd] = cols(hy); print('seed', sd, [f"{c:.2f}/{w:.2f}" for c, w in per[sd]], flush=True)
print("\nROW 8 (mean +/- sample std over 3 seeds), format CER / WER per column:")
cells = []
for j in range(5):
    cm = st.mean(per[s][j][0] for s in seeds); cs = st.stdev(per[s][j][0] for s in seeds)
    wm = st.mean(per[s][j][1] for s in seeds); ws = st.stdev(per[s][j][1] for s in seeds)
    cells.append(f"{cm:.1f}±{cs:.1f} / {wm:.1f}±{ws:.1f}"); print("  ", cells[-1])
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
def per_utt(row):
    o = []
    for u in ids:
        r = re.sub(r"\s", "", normalize(ref_of[u])); h = re.sub(r"\s", "", normalize(systems[row][u])); o.append((lev(r, h), len(r)))
    return o
cache = {k: per_utt(k) for k in [3, 4, *seeds]}
corp = lambda p, idx: 100 * sum(p[i][0] for i in idx) / sum(p[i][1] for i in idx)
n = len(ids); B = 10000; res = {}
for sd in seeds:
    for ref in (3, 4):
        random.seed(sd * 10 + ref); d = []
        for _ in range(B):
            idx = [random.randrange(n) for _ in range(n)]; d.append(corp(cache[sd], idx) - corp(cache[ref], idx))
        d.sort(); pt = corp(cache[sd], range(n)) - corp(cache[ref], range(n)); res[f"seed{sd}_vs_row{ref}"] = (pt, d[int(.025*B)], d[int(.975*B)])
        print(f"seed {sd} vs row {ref}: dCER={pt:+.2f} 95% CI [{d[int(.025*B)]:+.2f}, {d[int(.975*B)]:+.2f}]", flush=True)
json.dump({"row8_cells": cells, "bootstrap": res}, open(Path(os.environ.get("JOPARA_OUT", Path(__file__).resolve().parent)) / "row8_seeds.json", "w"))
print("done")
