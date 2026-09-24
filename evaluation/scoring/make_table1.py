#!/usr/bin/env python3
"""Paper Table 1 (rows 1-9), scored exactly as rescore_table1.py does (same normalize(), same sclite,
same 150-utterance references), plus per-utterance error counts.

Outputs (to $JOPARA_OUT, default: results/table1/ in this repo):
  table1.md                    the table as printed in the paper (CER / WER, %)
  table1.json                  same numbers, machine-readable (rows 8-9: per-seed values too)
  per_utterance_scores.csv     utt_id, subset, row, char/word sub/del/ins/ref counts -- no text

Summing a row's per-utterance counts over a subset reproduces that cell's corpus-level CER/WER.
Needs the hypothesis files and references ($JOPARA_ROOT, original project layout); they are not
shipped in this repo (see data/README.md).
"""
import csv, json, os, statistics as st, tempfile
from pathlib import Path
ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("JOPARA_OUT", HERE.parents[1] / "results" / "table1"))
src = open(HERE / "rescore_table1.py", encoding="utf-8").read().split("results, report = {}, []")[0]
ns = {"__name__": "rs", "__file__": str(Path(__file__).resolve().parent / "rescore_table1.py")}; exec(compile(src, "rs", "exec"), ns)
ids, ref_of, subset_of, systems, normalize = ns["ids"], ns["ref_of"], ns["subset_of"], ns["systems"], ns["normalize"]
SUBSETS, from_eval_csvs, from_branch, jl = ns["SUBSETS"], ns["from_eval_csvs"], ns["from_branch"], ns["jl"]
from metrics import score_with_sclite  # noqa: E402  (on sys.path via rescore_table1.py)

HP = ROOT / "results/June_15_Multitask_Audio_LLM/jopara_mole_asr/hypothesis_packs"
SEEDS = (13, 42, 123)
NAMES = {"1": "Whisper-large-v3, automatic language choice", "2": "MMS-1B-all, Guaraní adapter (no auto-detect)",
         "3": "OmniASR, auto-detect mode (baseline)", "4a": "OmniASR, Guaraní language mode (grn_Latn)",
         "4b": "OmniASR, Paraguayan Guaraní language mode (gug_Latn)", "4c": "OmniASR, Spanish language mode (spa_Latn)",
         "5": "Spanish adapter", "6": "Guaraní adapter", "7": "Code-switching adapter",
         "8": "Mixture of experts over all three adapters", "9": "Mixture of experts, code-switching adapter left empty"}
hyps = {"1": systems[1], "2": systems[2], "3": systems[3], "4a": systems[4],
        "4b": from_eval_csvs(ROOT / "results/June_09_baseline/gug_Latn"), "4c": from_branch("Z2_base_spa"),
        "5": systems[5], "6": systems[6], "7": systems[7]}
for sd in SEEDS:
    hyps[f"8/seed{sd}"] = {r["utt_id"]: r["hyp_jmole"] for r in jl(HP / f"row8_seed{sd}_a2_test_hypotheses.jsonl")}
    hyps[f"9/seed{sd}"] = {r["utt_id"]: r["hyp_jmole"] for r in jl(HP / f"a2_nullcs_seed{sd}_a2_test_hypotheses.jsonl")}


def per_utt(hy):
    missing = [u for u in ids if u not in hy]
    assert not missing, f"missing {len(missing)} utterances"
    refs, hs = [normalize(ref_of[u]) for u in ids], [normalize(hy[u]) for u in ids]
    with tempfile.TemporaryDirectory(prefix="table1_") as tmp:
        c = score_with_sclite(refs, hs, ids, Path(tmp) / "chr", level="char")["per_utt"]
        w = score_with_sclite(refs, hs, ids, Path(tmp) / "wrd", level="word")["per_utt"]
    return {u: (c[u], w[u]) for u in ids}


def rate(counts, level):
    k = 0 if level == "char" else 1
    err = sum(x[k]["sub"] + x[k]["del"] + x[k]["ins"] for x in counts)
    return 100 * err / sum(x[k]["n_ref"] for x in counts)


def cells(pu):
    out = []
    for s in SUBSETS + ["overall"]:
        sel = [pu[u] for u in ids if s == "overall" or subset_of[u] == s]
        out.append((rate(sel, "char"), rate(sel, "word")))
    return out


OUT.mkdir(parents=True, exist_ok=True)
scores, per = {}, {}
with open(OUT / "per_utterance_scores.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["row", "utt_id", "subset", "char_sub", "char_del", "char_ins", "char_ref",
                 "word_sub", "word_del", "word_ins", "word_ref"])
    for row, hy in hyps.items():
        pu = per_utt(hy); per[row] = cells(pu)
        for u in ids:
            c, w = pu[u]
            wr.writerow([row, u, subset_of[u], c["sub"], c["del"], c["ins"], c["n_ref"],
                         w["sub"], w["del"], w["ins"], w["n_ref"]])
        print(row, f"overall {per[row][4][0]:.1f} / {per[row][4][1]:.1f}", flush=True)

for row in ["1", "2", "3", "4a", "4b", "4c", "5", "6", "7"]:
    scores[row] = [f"{c:.1f} / {w:.1f}" for c, w in per[row]]
for row in ["8", "9"]:
    runs = [per[f"{row}/seed{sd}"] for sd in SEEDS]
    scores[row] = [f"{st.mean(r[j][0] for r in runs):.1f}±{st.stdev(r[j][0] for r in runs):.1f} / "
                   f"{st.mean(r[j][1] for r in runs):.1f}±{st.stdev(r[j][1] for r in runs):.1f}" for j in range(5)]

(OUT / "table1.json").write_text(json.dumps({"columns": SUBSETS + ["overall"], "rows": scores, "names": NAMES,
                                             "per_seed": {k: v for k, v in per.items() if "/seed" in k}},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
lines = ["# Table 1: CER / WER (%) on the 150-utterance test set, corpus-level sclite, no post-processing", "",
         "Rows 8 and 9: mean ± sample standard deviation over router seeds 13, 42, 123.", "",
         "| Row | System | Guaraní | Inter-sentential | Jehe'a | Spanish | Overall |", "|---|---|---|---|---|---|---|"]
for row in NAMES:  # paper column order: Guaraní, inter-sentential (mixed), jehe'a, Spanish, overall
    lines.append(f"| {row} | {NAMES[row]} | " + " | ".join(scores[row]) + " |")
(OUT / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT)
