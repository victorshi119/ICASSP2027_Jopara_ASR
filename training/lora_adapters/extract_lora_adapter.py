#!/usr/bin/env python3
"""
Standalone LoRA-adapter extractor for multi-seed runs (rebuttal #6, 2026-09-12).

Fixes a bug in lora_7b_spa/train_lora.py's save_lora_adapters_from_checkpoint():
that function globs checkpoint_dir.glob("step_*.tmp") directly under the run's
output_dir, which never matches this recipe's real layout
(<output_dir>/ws_*/checkpoints/step_<N>/model/pp_00/tp_00/sdp_00.pt, no ".tmp"
suffix). It then falls back to sorting *every* .pt file under output_dir by
mtime and takes the single most-recent one -- which can just as easily be the
tiny optimizer shard (.../optimizer/pp_00/tp_00/sdp_00.pt, ~tens of MB) as the
~31GB model shard, depending on write order. When it picks the optimizer
shard, none of its keys contain "lora_", so extraction silently fails with
"Could not extract LoRA state from checkpoints; check step_*.tmp layout."

This script instead explicitly globs for MODEL shards only
(.../checkpoints/step_<N>/model/pp_*/tp_*/sdp_*.pt), picks the highest step
number found (i.e. the true "latest" checkpoint, matching the same "latest,
not best" semantics train_lora.py documents/intends), and extracts only keys
containing "lora_".

Usage:
  python extract_lora_adapter.py --run-dir <output_dir> [--output <path>] [--step N]

  --step N forces a specific step instead of auto-picking the highest.

Do NOT run on a login node: loading a shard is a ~30-40GB single torch.load
into CPU RAM. Submit via run_extract_lora_adapters.sbatch (general partition).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def find_model_shards(run_dir: Path) -> dict[int, Path]:
    """Map step number -> model shard path, for every step with a shard on disk."""
    out: dict[int, Path] = {}
    for p in run_dir.glob("ws_*/checkpoints/step_*/model/pp_*/tp_*/sdp_*.pt"):
        step_dir = p.parents[3]  # .../checkpoints/step_<N>
        m = re.match(r"step_(\d+)$", step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        # If multiple ws_* dirs exist (e.g. after a resume), prefer the one
        # whose shard was written most recently for a given step number.
        if step not in out or p.stat().st_mtime > out[step].stat().st_mtime:
            out[step] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--step", type=int, default=None,
                     help="Force this step instead of auto-picking the highest available.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    output = args.output or (run_dir / "lora_adapters.pt")

    shards = find_model_shards(run_dir)
    if not shards:
        print(f"ERROR: no model shards found under {run_dir}/ws_*/checkpoints/step_*/model/**/sdp_*.pt")
        return 1

    step = args.step if args.step is not None else max(shards)
    if step not in shards:
        print(f"ERROR: requested step {step} not found; available steps: {sorted(shards)}")
        return 1
    ckpt_path = shards[step]

    print(f"Run dir:      {run_dir}")
    print(f"Steps found:  {sorted(shards)}")
    print(f"Using step:   {step}")
    print(f"Model shard:  {ckpt_path}  ({ckpt_path.stat().st_size / 1e9:.2f} GB)")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        print(f"ERROR: loaded object is {type(state)}, not a dict; cannot filter LoRA keys.")
        return 1

    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if not lora_state:
        sample_keys = list(state.keys())[:10]
        print(f"ERROR: no keys containing 'lora_' found. Sample keys: {sample_keys}")
        return 1

    n_params = sum(v.numel() for v in lora_state.values() if torch.is_tensor(v))
    print(f"Extracted {len(lora_state)} LoRA tensors, {n_params:,} parameters.")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state, output)
    print(f"Saved -> {output}  ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
