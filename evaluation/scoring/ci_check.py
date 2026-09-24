#!/usr/bin/env python3
"""0921: check the Results-section CI claims. Paired bootstrap of corpus CER (raw, sclite-style CER: spaces removed),
(a) resampling utterances (as in the paper), (b) resampling whole recording sessions (cluster bootstrap),
and per-category contrasts against the MATCHED no-adapter decoding mode."""
import os
import json, random, re, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
src = open(Path(__file__).resolve().parent / "rescore_table1.py", encoding="utf-8").read().split("results, report = {}, []")[0]
ns = {"__name__": "rs", "__file__": str(Path(__file__).resolve().parent / "rescore_table1.py")}; exec(compile(src, "rs", "exec"), ns)
ids, ref_of, systems, normalize, jl = ns["ids"], ns["ref_of"], ns["systems"], ns["normalize"], ns["jl"]
man = {r["utt_id"]: r for r in jl(ROOT / "moe_lora/manifests/v2_test_moe_manifest.jsonl")}
systems["4c"] = {r["utt_id"]: r["hypothesis"] for r in jl(ROOT / "moe_lora/predictions/all_branches/Z2_base_spa.jsonl")}
systems["4a"] = systems[4]
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
P = {}
for k in (3, "4a", "4c", 5, 6, 7, 8):
    P[k] = [(lev(re.sub(r"\s", "", normalize(ref_of[u])), re.sub(r"\s", "", normalize(systems[k][u]))),
             len(re.sub(r"\s", "", normalize(ref_of[u])))) for u in ids]
sess = [man[u]["session"] for u in ids]; cat = [man[u]["subset"] for u in ids]
print("sessions in test set:", len(set(sess)), "| utterances per session (max/median):",
      max(sess.count(s) for s in set(sess)), sorted(sess.count(s) for s in set(sess))[len(set(sess))//2])
def cer(p, idx): 
    d = sum(p[i][1] for i in idx); return 100 * sum(p[i][0] for i in idx) / d if d else float("nan")
def boot(a, b, pool, cluster, B=10000, seed=0):
    rnd = random.Random(seed); out = []
    if cluster:
        by = defaultdict(list)
        for i in pool: by[sess[i]].append(i)
        keys = list(by)
    for _ in range(B):
        if cluster: idx = [i for s in (rnd.choice(keys) for _ in keys) for i in by[s]]
        else: idx = [rnd.choice(pool) for _ in pool]
        out.append(cer(P[a], idx) - cer(P[b], idx))
    out.sort(); return out[int(.025 * B)], out[int(.975 * B)]
CATS = {"overall": list(range(len(ids)))}
for c in ("test_guarani_only", "test_mixed", "test_jehea", "test_spanish_only"):
    CATS[c] = [i for i in range(len(ids)) if cat[i] == c]
contrasts = [("4a", 3, "overall"), (7, 3, "overall"), (7, "4c", "overall"), (7, 3, "test_mixed"), (7, 3, "test_jehea"),
             (7, "4c", "test_mixed"), (7, "4c", "test_jehea"), (6, "4a", "test_guarani_only"), (6, "4a", "test_mixed"),
             (6, 3, "test_guarani_only"), (5, "4c", "test_spanish_only"), (5, 3, "test_spanish_only"), (8, 3, "overall"), (8, "4a", "overall")]
rows = []
for a, b, c in contrasts:
    pool = CATS[c]; pt = cer(P[a], pool) - cer(P[b], pool)
    lo, hi = boot(a, b, pool, False); clo, chi = boot(a, b, pool, True)
    flag = lambda l, h: "excl0" if (l > 0 or h < 0) else "incl0"
    line = f"row {a} - row {b} on {c:18s} (n={len(pool):3d}): d={pt:+.2f}  utt CI [{lo:+.2f},{hi:+.2f}] {flag(lo,hi)} | session-cluster CI [{clo:+.2f},{chi:+.2f}] {flag(clo,chi)}"
    print(line, flush=True); rows.append(line)
(Path(os.environ.get("JOPARA_OUT", Path(__file__).resolve().parent)) / "ci_check.txt").write_text("\n".join(rows) + "\n")
