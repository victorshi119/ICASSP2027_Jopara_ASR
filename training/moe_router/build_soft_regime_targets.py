#!/usr/bin/env python3
"""
Build per-utterance soft regime target distributions for MoLE-ASR training.

Reads manifests (which have v2_spa_count, v2_grn_count, v2_jha_count) and
writes a JSONL with one soft [p_spa, p_cs, p_grn] vector per utterance.

These soft targets replace hard one-hot labels during router KL training.

Outputs (in jopara_mole_asr/data/):
  soft_regime_targets_train.jsonl
  soft_regime_targets_dev.jsonl
  soft_regime_targets_test.jsonl   (only for final eval — do not tune on this)

Usage:
  $OMNI_PY scripts/build_soft_regime_targets.py
  $OMNI_PY scripts/build_soft_regime_targets.py --splits train dev
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJ_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))  # data root (original project layout)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "lora_adapters"))

BASE   = _PROJ_ROOT
J14    = BASE / "results/June_14_Multitask_Audio_LLM"
J15    = BASE / "results/June_15_Multitask_Audio_LLM"
OUT    = J15 / "jopara_mole_asr/data"

from model.soft_regime_router import make_soft_target


MANIFEST_MAP = {
    "train": [
        J14 / "manifests/train_cegpa_spa.jsonl",
        J14 / "manifests/train_cegpa_cs.jsonl",
        J14 / "manifests/train_grn_cv.jsonl",
    ],
    "dev": [
        J14 / "manifests/dev_spa.jsonl",
        J14 / "manifests/dev_cs.jsonl",
        J14 / "manifests/dev_grn.jsonl",
    ],
    "test": [
        BASE / "moe_lora/manifests/v2_test_moe_manifest.jsonl",
    ],
}


def load_manifest(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping")
            continue
        chunk = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows.extend(chunk)
        print(f"  Loaded {len(chunk):4d} rows from {p.name}")
    return rows


def build_targets(rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    regime_counts: Counter = Counter()

    for row in rows:
        regime = row.get("regime", "")
        if regime not in ("spa", "cs", "grn"):
            # Normalise legacy category names
            cat = row.get("v2_category", "")
            if cat == "spanish_only":
                regime = "spa"
            elif cat == "guarani_only":
                regime = "grn"
            elif cat in ("mixed", "jehe'a"):
                regime = "cs"
            else:
                regime = "cs"   # fallback

        spa_c = int(row.get("v2_spa_count",   0) or 0)
        grn_c = int(row.get("v2_grn_count",   0) or 0)
        jha_c = int(row.get("v2_jha_count",   0) or 0)

        soft = make_soft_target(regime, spa_c, grn_c, jha_c)
        regime_counts[regime] += 1

        # audio_stem gives the segment-level ID (tr-008_01922) derived from the
        # audio filename. Some embedding files use this format instead of the
        # manifest utt_id (which is session-only, e.g. tr-008). We write both so
        # the router training script can match either key.
        audio_path = row.get("audio_path", "")
        audio_stem = Path(audio_path).stem if audio_path else row["utt_id"]

        records.append({
            "utt_id":      row["utt_id"],
            "audio_stem":  audio_stem,
            "regime":      regime,
            "v2_spa_count": spa_c,
            "v2_grn_count": grn_c,
            "v2_jha_count": jha_c,
            "soft_target": soft,   # [p_spa, p_cs, p_grn]
        })

    return records, dict(regime_counts)


def print_stats(records: list[dict], split: str) -> None:
    import numpy as np
    by_regime: dict[str, list] = defaultdict(list)
    for r in records:
        by_regime[r["regime"]].append(r["soft_target"])

    print(f"\n  Split: {split}  ({len(records)} utterances)")
    print(f"  {'Regime':<8}  {'N':>5}  {'mean_p_spa':>10}  {'mean_p_cs':>10}  {'mean_p_grn':>10}")
    for regime in ("spa", "cs", "grn"):
        vecs = by_regime.get(regime, [])
        if not vecs:
            continue
        arr = np.array(vecs)
        print(f"  {regime:<8}  {len(vecs):>5}  "
              f"{arr[:,0].mean():>10.3f}  "
              f"{arr[:,1].mean():>10.3f}  "
              f"{arr[:,2].mean():>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+",
        choices=["train", "dev", "test"],
        default=["train", "dev"],
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n── Building soft targets: {split} ──")
        rows = load_manifest(MANIFEST_MAP[split])
        if not rows:
            print(f"  No rows loaded for {split} — skipping")
            continue

        records, dist = build_targets(rows)
        print(f"  Regime distribution: {dist}")
        print_stats(records, split)

        out_path = OUT / f"soft_regime_targets_{split}.jsonl"
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} targets → {out_path}")


if __name__ == "__main__":
    main()
