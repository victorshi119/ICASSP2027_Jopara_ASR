#!/usr/bin/env python3
"""
Jopara ASR metrics via SCTK sclite.

CER (corpus-level, no-space) is the primary metric.
WER (corpus-level, word-level) is the secondary metric.
Both macro-averaged variants are also reported for transparency.

CLI interface (drop-in for the old home-grown metrics.py):
  python metrics.py --ref ref.txt --hyp hyp.txt [--subset name] [--output metrics.json]

Sclite binary: /N/slate/wencshi/miniforge3/bin/sclite (conda-forge sctk 2.4.12)
CER convention: strip spaces, then treat each Unicode character as a token.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

SCLITE = shutil.which("sclite") or "/N/slate/wencshi/miniforge3/bin/sclite"


# ---------------------------------------------------------------------------
# Normalization — identical to the old metrics.py so scores are comparable
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Collapse whitespace only. Preserve case and diacritics (Guaraní/Spanish)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def to_char_tokens(text: str) -> str:
    """Strip spaces, then join every Unicode character with a single space.

    'hola mundo' → 'h o l a m u n d o'

    This is the standard ASR CER tokenisation used with sclite.
    """
    return " ".join(list(text.replace(" ", "")))


# ---------------------------------------------------------------------------
# sclite helpers
# ---------------------------------------------------------------------------

def _write_trn(texts: list[str], ids: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for text, uid in zip(texts, ids):
            clean = " ".join(str(text).split())
            f.write(f"{clean} ({uid})\n")


def _run_sclite(ref_trn: Path, hyp_trn: Path, work_dir: Path) -> Tuple[Path, bool]:
    """Run sclite and return (sgml_path, success)."""
    sgml_out = work_dir / (hyp_trn.name + ".sgml")
    try:
        result = subprocess.run(
            [
                SCLITE,
                "-r", str(ref_trn), "trn",
                "-h", str(hyp_trn), "trn",
                "-i", "wsj",
                "-o", "sgml",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(work_dir),
        )
        # sclite writes {hyp_trn}.sgml next to the hyp file
        generated = Path(str(hyp_trn) + ".sgml")
        if generated.exists():
            shutil.move(str(generated), str(sgml_out))
            return sgml_out, True
        # Fallback: look in work_dir
        candidate = work_dir / (hyp_trn.name + ".sgml")
        if candidate.exists():
            return candidate, True
        sys.stderr.write(f"sclite failed:\n{result.stderr}\n")
        return sgml_out, False
    except Exception as e:
        sys.stderr.write(f"sclite subprocess error: {e}\n")
        return sgml_out, False


def _parse_sgml_corpus(sgml_path: Path) -> Dict:
    """
    Parse sclite SGML and return corpus-level totals:
      corr, sub, del_, ins, n_ref, n_sent
    """
    corr = sub = del_ = ins = n_sent = 0
    if not sgml_path.exists():
        return {}
    content = sgml_path.read_text(encoding="utf-8", errors="replace")
    # Each <PATH> block has the per-utterance alignment
    for block_data in re.findall(r"<PATH[^>]*>(.*?)</PATH>", content, re.DOTALL):
        n_sent += 1
        for segment in block_data.strip().split(":"):
            segment = segment.strip()
            if not segment:
                continue
            op = segment[0]
            if op == "C":
                corr += 1
            elif op == "S":
                sub += 1
            elif op == "D":
                del_ += 1
            elif op == "I":
                ins += 1
    n_ref = corr + sub + del_
    return {"corr": corr, "sub": sub, "del": del_, "ins": ins,
            "n_ref": n_ref, "n_sent": n_sent}


def _parse_sgml_per_utt(sgml_path: Path) -> Dict[str, Dict]:
    """Return per-utterance {uid: {sub, del, ins, n_ref}} from SGML."""
    results = {}
    if not sgml_path.exists():
        return results
    content = sgml_path.read_text(encoding="utf-8", errors="replace")
    for uid, block_data in re.findall(
        r'<PATH id="\(?([^"\)]+)\)?"[^>]*>(.*?)</PATH>', content, re.DOTALL
    ):
        corr = sub = del_ = ins = 0
        for segment in block_data.strip().split(":"):
            segment = segment.strip()
            if not segment:
                continue
            op = segment[0]
            if op == "C":
                corr += 1
            elif op == "S":
                sub += 1
            elif op == "D":
                del_ += 1
            elif op == "I":
                ins += 1
        results[uid] = {"sub": sub, "del": del_, "ins": ins,
                        "n_ref": max(1, corr + sub + del_)}
    return results


# ---------------------------------------------------------------------------
# Corpus-level scoring via sclite
# ---------------------------------------------------------------------------

def score_with_sclite(
    refs: list[str],
    hyps: list[str],
    ids: list[str],
    work_dir: Path,
    level: str,          # "word" or "char"
) -> Dict:
    """
    Score refs vs hyps at word or character level using sclite.
    Returns corpus-level totals + per-utterance dict.
    """
    tag = level
    if level == "char":
        refs_tok = [to_char_tokens(r) for r in refs]
        hyps_tok = [to_char_tokens(h) for h in hyps]
    else:
        refs_tok = refs
        hyps_tok = hyps

    work_dir.mkdir(parents=True, exist_ok=True)
    ref_trn = work_dir / f"ref.{tag}.trn"
    hyp_trn = work_dir / f"hyp.{tag}.trn"
    _write_trn(refs_tok, ids, ref_trn)
    _write_trn(hyps_tok, ids, hyp_trn)

    sgml_path, ok = _run_sclite(ref_trn, hyp_trn, work_dir)
    if not ok:
        return {}

    corpus = _parse_sgml_corpus(sgml_path)
    per_utt = _parse_sgml_per_utt(sgml_path)
    corpus["per_utt"] = per_utt
    return corpus


# ---------------------------------------------------------------------------
# Macro-averaged fallback (pure Python, for per-utterance macro stats)
# ---------------------------------------------------------------------------

def _edit_counts_word(ref_words: list, hyp_words: list) -> tuple:
    n, m = len(ref_words), len(hyp_words)
    INF = 10 ** 9
    dp = [[(INF, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = (INF, 0, 0, 0)
            if i > 0:
                c, ai, ad, as_ = dp[i - 1][j]
                best = min(best, (c + 1, ai, ad + 1, as_), key=lambda x: x[0])
            if j > 0:
                c, ai, ad, as_ = dp[i][j - 1]
                best = min(best, (c + 1, ai + 1, ad, as_), key=lambda x: x[0])
            if i > 0 and j > 0:
                c, ai, ad, as_ = dp[i - 1][j - 1]
                sub = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                best = min(best, (c + sub, ai, ad, as_ + sub), key=lambda x: x[0])
            dp[i][j] = best
    return dp[n][m][1], dp[n][m][2], dp[n][m][3], n


def _macro_stats(refs: list[str], hyps: list[str]) -> Dict:
    wer_sum = cer_sum = 0.0
    n_ref_w_total = n_ref_c_total = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        rc = r.replace(" ", "")
        hc = h.replace(" ", "")
        ni_w, nd_w, ns_w, nw = _edit_counts_word(rw, hw)
        ni_c, nd_c, ns_c, nc = _edit_counts_word(list(rc), list(hc))
        wer_sum += (ni_w + nd_w + ns_w) / nw if nw else 0.0
        cer_sum += (ni_c + nd_c + ns_c) / nc if nc else 0.0
        n_ref_w_total += nw
        n_ref_c_total += nc
    n = len(refs)
    return {
        "wer_macro": wer_sum / n if n else 0.0,
        "cer_macro": cer_sum / n if n else 0.0,
        "n_ref_words": n_ref_w_total,
        "n_ref_chars": n_ref_c_total,
    }


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def compute_metrics(refs: list[str], hyps: list[str]) -> Dict:
    """
    Compute CER (primary) and WER (secondary) using sclite, plus macro averages.
    Falls back to pure-Python if sclite is unavailable.
    """
    n = len(refs)
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    ids = [f"utt{i:06d}" for i in range(1, n + 1)]

    macro = _macro_stats(refs, hyps)

    if not os.path.isfile(SCLITE):
        sys.stderr.write(
            f"WARNING: sclite not found at {SCLITE}; falling back to Python DP.\n"
            "  Install with: conda install -c conda-forge sctk\n"
        )
        return {
            "n_utterances": n,
            **macro,
            # Primary / secondary with fallback values
            "cer_corpus": macro["cer_macro"],
            "wer_corpus": macro["wer_macro"],
            # Backward compat
            "cer_overall": macro["cer_macro"],
            "wer_overall": macro["wer_macro"],
            "sclite": False,
        }

    with tempfile.TemporaryDirectory(prefix="jopara_sclite_") as tmp:
        work = Path(tmp)

        wrd = score_with_sclite(refs, hyps, ids, work / "wrd", level="word")
        cha = score_with_sclite(refs, hyps, ids, work / "cha", level="char")

    def _rate(tot: Dict, key_num: str) -> float:
        n_ref = tot.get("n_ref", 1) or 1
        return (tot.get("sub", 0) + tot.get("del", 0) + tot.get("ins", 0)) / n_ref

    cer_corpus = _rate(cha, "n_ref") if cha else macro["cer_macro"]
    wer_corpus = _rate(wrd, "n_ref") if wrd else macro["wer_macro"]

    result = {
        "n_utterances": n,
        "n_ref_words": macro["n_ref_words"],
        "n_ref_chars": macro["n_ref_chars"],
        # Primary
        "cer_corpus": round(cer_corpus, 6),
        "cer_sub":    cha.get("sub", 0),
        "cer_del":    cha.get("del", 0),
        "cer_ins":    cha.get("ins", 0),
        "cer_n_ref":  cha.get("n_ref", 0),
        # Secondary
        "wer_corpus": round(wer_corpus, 6),
        "wer_sub":    wrd.get("sub", 0),
        "wer_del":    wrd.get("del", 0),
        "wer_ins":    wrd.get("ins", 0),
        "wer_n_ref":  wrd.get("n_ref", 0),
        # Macro averages (kept for comparison and backward compat)
        "cer_macro":  round(macro["cer_macro"], 6),
        "wer_macro":  round(macro["wer_macro"], 6),
        # Backward-compat keys (old callers expect these)
        "cer_overall": round(cer_corpus, 6),
        "wer_overall": round(wer_corpus, 6),
        "sclite": True,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Jopara ASR metrics (sclite). CER primary, WER secondary."
    )
    p.add_argument("--ref",    type=Path, required=True, help="Reference file (one line/utterance)")
    p.add_argument("--hyp",    type=Path, required=True, help="Hypothesis file (one line/utterance)")
    p.add_argument("--subset", type=str,  default="overall")
    p.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = p.parse_args()

    if not args.ref.exists():
        sys.exit(f"ERROR: ref file not found: {args.ref}")
    if not args.hyp.exists():
        sys.exit(f"ERROR: hyp file not found: {args.hyp}")

    refs = args.ref.read_text(encoding="utf-8").splitlines()
    hyps = args.hyp.read_text(encoding="utf-8").splitlines()

    if len(refs) != len(hyps):
        sys.exit(f"ERROR: length mismatch: ref={len(refs)} hyp={len(hyps)}")

    metrics = compute_metrics(refs, hyps)

    # Update or create output JSON
    if args.output.exists():
        data = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        data = {}
    data[args.subset] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Human-readable summary — CER first
    cer_pct = metrics["cer_corpus"] * 100
    wer_pct = metrics["wer_corpus"] * 100
    cer_mac = metrics["cer_macro"] * 100
    wer_mac = metrics["wer_macro"] * 100
    n       = metrics["n_utterances"]
    print(f"[{args.subset}]  n={n}")
    print(f"  CER (corpus, PRIMARY) : {cer_pct:.2f}%   "
          f"(S={metrics['cer_sub']} D={metrics['cer_del']} I={metrics['cer_ins']} "
          f"N={metrics['cer_n_ref']})")
    print(f"  WER (corpus)          : {wer_pct:.2f}%   "
          f"(S={metrics['wer_sub']} D={metrics['wer_del']} I={metrics['wer_ins']} "
          f"N={metrics['wer_n_ref']})")
    print(f"  CER macro-avg         : {cer_mac:.2f}%")
    print(f"  WER macro-avg         : {wer_mac:.2f}%")
    print(f"  Written → {args.output}")


if __name__ == "__main__":
    main()
