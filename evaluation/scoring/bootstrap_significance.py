#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals / significance tests for the ICASSP 2026
rebuttal, addressing reviewer comment #6 ("Statistical reliability of the
reported improvements").

For a pair of systems (A = baseline, B = candidate) evaluated on the SAME
150-utterance v2 test set, this script:

  1. Gets per-utterance edit counts (sub, del, ins, n_ref) from sclite at both
     char level (CER) and word level (WER), for A and for B, using the exact
     same normalization/tokenization as metrics.py (so numbers reproduce the
     paper's corpus-level sclite figures).
  2. Runs a paired bootstrap over utterances (Koehn 2004-style): resample
     utterance indices with replacement B_BOOT times, recompute the CORPUS
     (micro-averaged) CER/WER for A and B on each resample, and collect the
     distribution of delta = CER_B - CER_A (and WER_B - WER_A).
  3. Reports point estimates, 95% percentile CIs for CER_A, CER_B, and the
     paired delta, plus a two-sided bootstrap p-value for delta != 0.
  4. Repeats per subset (Guarani-only n=25, Spanish-only n=25, Mixed n=67,
     Jehe'a n=33) and overall (n=150).

Usage:
    python bootstrap_significance.py --config configs.json --out results/

Each entry in the JSON config list is a "comparison":
    {
      "name": "P_morph_vs_A2NullCS",
      "file": "/path/to/corrected_hyps.jsonl",
      "id_key": "utt_id", "subset_key": "v2_category",
      "ref_key": "ref", "a_key": "hyp_orig", "b_key": "hyp_corrected",
      "system_a": "A2-NullCS", "system_b": "P_morph",
      "fill_missing_from": null   # or path to a 150-row file to pad subsets absent here (e.g. Spanish passthrough)
    }
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from postprocess_validation import postprocess as _postprocess  # noqa: E402

# The official pipeline scores postprocess(text) -> normalize() -> sclite (see
# results/June_24_Experiments/agglutinative_ger/scripts/evaluate_full150.py).
# Apply the same normalization here so point estimates reproduce the paper's
# reported corpus CER/WER exactly (raw un-normalized text scores ~0.7pp higher).

def prep(text):
    return _postprocess(str(text or ""))

N_BOOT = 10000
SEED = 13
CI_LO, CI_HI = 2.5, 97.5  # percentile bounds for 95% CI


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pairs(cfg):
    """Return list of dicts {utt_id, subset, ref, hyp_a, hyp_b}, 150 rows."""
    rows = load_jsonl(cfg["file"])
    out = {}
    for r in rows:
        uid = r[cfg["id_key"]]
        out[uid] = {
            "utt_id": uid,
            "subset": r[cfg["subset_key"]],
            "ref": r[cfg["ref_key"]],
            "hyp_a": r[cfg["a_key"]],
            "hyp_b": r[cfg["b_key"]],
        }
    if cfg.get("fill_missing_from"):
        fill_rows = load_jsonl(cfg["fill_missing_from"])
        fkeys = cfg.get("fill_keys", cfg)
        for r in fill_rows:
            uid = r[fkeys.get("fill_id_key", cfg["id_key"])]
            if uid not in out:
                # passthrough: hyp unchanged by the correction system
                out[uid] = {
                    "utt_id": uid,
                    "subset": r[fkeys.get("fill_subset_key", cfg["subset_key"])],
                    "ref": r[fkeys.get("fill_ref_key", "ref")],
                    "hyp_a": r[fkeys.get("fill_hyp_key", "hyp")],
                    "hyp_b": r[fkeys.get("fill_hyp_key", "hyp")],
                }
    pairs = list(out.values())
    return pairs


def sclite_errors(refs, hyps, ids, level):
    """Return {uid: {sub,del,ins,n_ref}} via sclite (char or word level)."""
    refs_n = [M.normalize(r) for r in refs]
    hyps_n = [M.normalize(h) for h in hyps]
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rebuttal_sclite_") as tmp:
        work = Path(tmp)
        result = M.score_with_sclite(refs_n, hyps_n, ids, work / level, level=level)
    if not result:
        raise RuntimeError(f"sclite scoring failed for level={level}")
    return result["per_utt"], result


def corpus_rate(err_arr, nref_arr, idx):
    num = sum(err_arr[i] for i in idx)
    den = sum(nref_arr[i] for i in idx)
    return num / den if den else 0.0


def paired_bootstrap(err_a, err_b, nref, n_boot=N_BOOT, seed=SEED):
    n = len(nref)
    rng = random.Random(seed)
    point_a = corpus_rate(err_a, nref, range(n))
    point_b = corpus_rate(err_b, nref, range(n))
    point_delta = point_b - point_a

    boot_a, boot_b, boot_delta = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ca = corpus_rate(err_a, nref, idx)
        cb = corpus_rate(err_b, nref, idx)
        boot_a.append(ca)
        boot_b.append(cb)
        boot_delta.append(cb - ca)

    boot_a.sort(); boot_b.sort(); boot_delta.sort()

    def pct(sorted_list, p):
        k = (len(sorted_list) - 1) * (p / 100.0)
        f, c = int(k), min(int(k) + 1, len(sorted_list) - 1)
        if f == c:
            return sorted_list[f]
        return sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f)

    ci_a = (pct(boot_a, CI_LO), pct(boot_a, CI_HI))
    ci_b = (pct(boot_b, CI_LO), pct(boot_b, CI_HI))
    ci_delta = (pct(boot_delta, CI_LO), pct(boot_delta, CI_HI))

    # two-sided bootstrap p-value (Koehn 2004): proportion of resamples on the
    # "wrong side" of zero, doubled, capped at 1.
    n_le0 = sum(1 for d in boot_delta if d <= 0)
    n_ge0 = sum(1 for d in boot_delta if d >= 0)
    p_value = 2.0 * min(n_le0, n_ge0) / n_boot
    p_value = min(p_value, 1.0)

    return {
        "n": n,
        "point_a": point_a, "point_b": point_b, "point_delta": point_delta,
        "ci_a": ci_a, "ci_b": ci_b, "ci_delta": ci_delta,
        "p_value": p_value,
        "significant_at_05": not (ci_delta[0] <= 0.0 <= ci_delta[1]),
    }


def run_comparison(cfg, n_boot):
    pairs = build_pairs(cfg)
    assert len(pairs) == 150, f"{cfg['name']}: expected 150 utterances, got {len(pairs)}"
    ids = [p["utt_id"] for p in pairs]
    refs = [prep(p["ref"]) for p in pairs]
    hyps_a = [prep(p["hyp_a"]) for p in pairs]
    hyps_b = [prep(p["hyp_b"]) for p in pairs]
    subsets = [p["subset"] for p in pairs]

    per_utt_char_a, _ = sclite_errors(refs, hyps_a, ids, "char")
    per_utt_char_b, _ = sclite_errors(refs, hyps_b, ids, "char")
    per_utt_word_a, _ = sclite_errors(refs, hyps_a, ids, "word")
    per_utt_word_b, _ = sclite_errors(refs, hyps_b, ids, "word")

    def arr(per_utt, key):
        return [per_utt[u][key] for u in ids]

    def errs(per_utt):
        return [per_utt[u]["sub"] + per_utt[u]["del"] + per_utt[u]["ins"] for u in ids]

    cer_err_a, cer_err_b = errs(per_utt_char_a), errs(per_utt_char_b)
    cer_nref = arr(per_utt_char_a, "n_ref")  # same ref -> same n_ref as per_utt_char_b (sanity-checked below)
    wer_err_a, wer_err_b = errs(per_utt_word_a), errs(per_utt_word_b)
    wer_nref = arr(per_utt_word_a, "n_ref")

    # sanity: n_ref must match between system A and B scoring (same reference text)
    cer_nref_b = arr(per_utt_char_b, "n_ref")
    wer_nref_b = arr(per_utt_word_b, "n_ref")
    assert cer_nref == cer_nref_b, f"{cfg['name']}: CER n_ref mismatch between systems"
    assert wer_nref == wer_nref_b, f"{cfg['name']}: WER n_ref mismatch between systems"

    subset_names = sorted(set(subsets)) + ["OVERALL"]
    out = {"name": cfg["name"], "system_a": cfg["system_a"], "system_b": cfg["system_b"],
           "subsets": {}}

    for sname in subset_names:
        if sname == "OVERALL":
            idx = list(range(len(ids)))
        else:
            idx = [i for i, s in enumerate(subsets) if s == sname]
        if not idx:
            continue
        sub_cer_a = [cer_err_a[i] for i in idx]
        sub_cer_b = [cer_err_b[i] for i in idx]
        sub_cer_nref = [cer_nref[i] for i in idx]
        sub_wer_a = [wer_err_a[i] for i in idx]
        sub_wer_b = [wer_err_b[i] for i in idx]
        sub_wer_nref = [wer_nref[i] for i in idx]

        res_cer = paired_bootstrap(sub_cer_a, sub_cer_b, sub_cer_nref, n_boot=n_boot, seed=SEED)
        res_wer = paired_bootstrap(sub_wer_a, sub_wer_b, sub_wer_nref, n_boot=n_boot, seed=SEED + 1)
        out["subsets"][sname] = {"cer": res_cer, "wer": res_wer}

    return out


def fmt_pct(x):
    return f"{x * 100:.2f}"


def print_report(result):
    print("=" * 100)
    print(f"{result['name']}: {result['system_a']} (A) vs {result['system_b']} (B), "
          f"paired bootstrap, N_boot={N_BOOT}, seed={SEED}")
    print("=" * 100)
    header = f"{'Subset':<14}{'n':>5}  {'CER_A':>8} {'CER_B':>8} {'ΔCER':>8}  {'95% CI (ΔCER)':>20}  {'p':>7}  {'sig?':>5}"
    print(header)
    for sname, d in result["subsets"].items():
        r = d["cer"]
        ci = f"[{fmt_pct(r['ci_delta'][0])}, {fmt_pct(r['ci_delta'][1])}]"
        sig = "YES" if r["significant_at_05"] else "no"
        print(f"{sname:<14}{r['n']:>5}  {fmt_pct(r['point_a']):>8} {fmt_pct(r['point_b']):>8} "
              f"{fmt_pct(r['point_delta']):>8}  {ci:>20}  {r['p_value']:>7.4f}  {sig:>5}")
    print()
    header = f"{'Subset':<14}{'n':>5}  {'WER_A':>8} {'WER_B':>8} {'ΔWER':>8}  {'95% CI (ΔWER)':>20}  {'p':>7}  {'sig?':>5}"
    print(header)
    for sname, d in result["subsets"].items():
        r = d["wer"]
        ci = f"[{fmt_pct(r['ci_delta'][0])}, {fmt_pct(r['ci_delta'][1])}]"
        sig = "YES" if r["significant_at_05"] else "no"
        print(f"{sname:<14}{r['n']:>5}  {fmt_pct(r['point_a']):>8} {fmt_pct(r['point_b']):>8} "
              f"{fmt_pct(r['point_delta']):>8}  {ci:>20}  {r['p_value']:>7.4f}  {sig:>5}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    configs = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    all_results = []
    report_lines = []
    import io, contextlib
    for cfg in configs:
        print(f"Running: {cfg['name']} ...", file=sys.stderr)
        res = run_comparison(cfg, n_boot=args.n_boot)
        all_results.append(res)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(res)
        text = buf.getvalue()
        print(text)
        report_lines.append(text)

    (args.out / "bootstrap_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    (args.out / "bootstrap_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Written -> {args.out / 'bootstrap_report.txt'}", file=sys.stderr)
    print(f"Written -> {args.out / 'bootstrap_results.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
