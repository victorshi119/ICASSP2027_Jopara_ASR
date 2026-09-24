#!/usr/bin/env python3
"""
LoRA fine-tuning for OmniASR LLM 7B v2 with lang=spa_Latn.
Thin wrapper: load A0–A4 config, inject LoRA via recipe, run fairseq2 trainer, save adapters + metadata.

Usage:
  PYTHONPATH="<OMNILINGUAL_ASR_ROOT>:<JoparaASR_v2>" python -m lora_7b_spa.train_lora \\
    --config config/a0_baseline.yaml --output-dir runs/A0 \\
    [--manifest-dir ...] [--dataset-name jopara_7b_regular_manifest] [--model-card omniASR_LLM_7B_v2]

Data: use data_prep.py to build manifest (train.tsv, train.wrd, dev.tsv, dev.wrd) with lang=spa_Latn.
The recipe uses dataset.name (dataset card) to resolve manifest path; ensure the card points to your manifest_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_recipe_yaml(
    base_recipe_path: Path,
    our_config: dict,
    dataset_name: str,
    model_card: str,
    seed: int | None = None,
) -> dict:
    with base_recipe_path.open("r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    recipe["model"] = {"name": model_card}
    recipe["dataset"]["name"] = dataset_name
    if seed is not None:
        # fairseq2's CommonSection.seed defaults to a hardcoded 2 (recipe/config.py) — every
        # run is otherwise identical (LoRA init, dropout, data shuffling all derive from it).
        # Set explicitly for multi-seed variance runs.
        recipe.setdefault("common", {})["seed"] = int(seed)
    tr = our_config.get("training", {})
    recipe["optimizer"]["config"] = {
        "lr": float(tr.get("learning_rate", 1e-4)),
        "weight_decay": float(tr.get("weight_decay", 0.01)),
        "betas": list(tr.get("adam_betas", [0.9, 0.98])),
        "eps": float(tr.get("adam_eps", 1e-8)),
    }
    ga = int(tr.get("gradient_accumulation_steps", 8))
    recipe["trainer"]["grad_accumulation"] = {"num_batches": ga}
    # Base recipe's _apply_trainable_param_patterns unfreezes matching params. Use decoder scope so base
    # finds them (pattern "lora_" matched 0 params, so base may see model before our inject; "llama_decoder" matches all decoder params including LoRA).
    recipe["trainer"]["trainable_param_patterns"] = ["llama_decoder"]
    max_steps = int(tr.get("max_steps", 4000))
    eval_every = int(tr.get("eval_every", 200))
    # Checkpoint cadence: config checkpoint.save_every or training.checkpoint_every, else eval_every, else 200.
    # Set to max_steps to save only at the end (saves disk); otherwise checkpoints are saved periodically for resume.
    ckpt_cfg = our_config.get("checkpoint", {})
    checkpoint_every = int(
        tr.get("checkpoint_every")
        or ckpt_cfg.get("save_every")
        or eval_every
        or 200
    )
    recipe["regime"] = {
        "num_steps": max_steps,
        "validate_after_n_steps": 0,
        "validate_every_n_steps": eval_every,
        "checkpoint_every_n_steps": min(checkpoint_every, max_steps),
        "publish_metrics_every_n_steps": max(1, eval_every // 4),
    }
    return recipe


def save_lora_adapters_from_checkpoint(
    checkpoint_dir: Path,
    output_path: Path,
) -> bool:
    """Load full checkpoint, extract LoRA params, save to output_path. Uses latest step_*.tmp."""
    import torch
    step_dirs = list(checkpoint_dir.glob("step_*.tmp"))
    if not step_dirs:
        step_files = sorted(checkpoint_dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not step_files:
            return False
        ckpt = step_files[0]
    else:
        def step_nr(p: Path) -> int:
            try:
                return int(p.name.replace("step_", "").replace(".tmp", ""))
            except ValueError:
                return -1
        latest = max(step_dirs, key=step_nr)
        model_pt = latest / "model/pp_00/tp_00/sdp_00.pt"
        if not model_pt.exists():
            model_pt = next(latest.rglob("sdp_00.pt"), None)
        if not model_pt or not model_pt.exists():
            return False
        ckpt = model_pt
    # weights_only=False: fairseq2 checkpoints use pickle protocol 5 / custom objects (PyTorch 2.6+ defaults to weights_only=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return False
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if not lora_state:
        return False
    torch.save(lora_state, output_path)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="LoRA fine-tune OmniASR LLM 7B v2 (fairseq2 recipe + LoRA).")
    p.add_argument("--config", type=Path, required=True, help="e.g. config/a0_baseline.yaml")
    p.add_argument("--output-dir", type=Path, required=True, help="Run dir: checkpoints, logs, lora_adapters.pt")
    p.add_argument("--manifest-dir", type=Path, default=None, help="Manifest dir (train.tsv, dev.tsv); dataset card must point here")
    p.add_argument("--dataset-name", type=str, default="jopara_7b_regular_manifest", help="Dataset card name in omnilingual-asr")
    p.add_argument("--model-card", type=str, default="omniASR_LLM_7B_v2")
    p.add_argument(
        "--load-from-checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (e.g. A0_gug step_2000 model sdp_00.pt) to continue fine-tuning; same LoRA config.",
    )
    p.add_argument("--list-modules-only", action="store_true", help="Print decoder Linear module names and exit")
    p.add_argument("--skip-train", action="store_true", help="Only write configs and run_info; do not run trainer")
    p.add_argument("--seed", type=int, default=None,
                   help="fairseq2 common.seed override (default: fairseq2's hardcoded 2, i.e. reruns are identical)")
    args = p.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write config copy and LoRA config for the recipe
    (args.output_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    lora_cfg = config.get("lora", {})
    lora_config_path = args.output_dir / "lora_config.json"
    lora_config_path.write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")

    if args.list_modules_only:
        try:
            script_dir = Path(__file__).resolve().parent
            sys.path.insert(0, str(script_dir))
            from list_decoder_modules import print_decoder_linear_modules
            print_decoder_linear_modules(args.model_card, None)
        except Exception as e:
            print("Set OMNILINGUAL_ASR_ROOT / PYTHONPATH and run list_decoder_modules.py", e)
        return 0

    # Build recipe YAML
    base_recipe = Path(__file__).resolve().parent / "config" / "recipe_base_lora.yaml"
    recipe_dict = build_recipe_yaml(base_recipe, config, args.dataset_name, args.model_card, seed=args.seed)
    recipe_yaml_path = args.output_dir / "recipe_lora.yaml"
    with recipe_yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(recipe_dict, f, default_flow_style=False, allow_unicode=True)

    run_info = {
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "manifest_dir": str(args.manifest_dir) if args.manifest_dir else None,
        "dataset_name": args.dataset_name,
        "model_card": args.model_card,
        "load_from_checkpoint": str(args.load_from_checkpoint) if args.load_from_checkpoint else None,
        "seed": args.seed,
        "lora": lora_cfg,
        "training": config.get("training", {}),
        "recipe_yaml": str(recipe_yaml_path),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    if args.skip_train:
        print("Wrote config, lora_config.json, recipe_lora.yaml, run_info.json. Skipping training (--skip-train).")
        return 0

    # Run the recipe with LoRA (same process so LORA_CONFIG_PATH is set)
    os.environ["LORA_CONFIG_PATH"] = str(lora_config_path)
    if args.load_from_checkpoint is not None:
        ckpt_path = args.load_from_checkpoint.resolve()
        if not ckpt_path.exists():
            print("Error: --load-from-checkpoint path does not exist:", ckpt_path, file=sys.stderr)
            return 1
        os.environ["LOAD_FROM_CHECKPOINT"] = str(ckpt_path)
    cmd = [
        sys.executable, "-m", "lora_7b_spa.run_recipe_lora",
        str(args.output_dir),
        "--config-file", str(recipe_yaml_path),
    ]
    print("Running:", " ".join(cmd))
    ret = subprocess.run(cmd, env=os.environ.copy())
    if ret.returncode != 0:
        return ret.returncode

    # Save LoRA adapters from latest checkpoint
    adapter_path = args.output_dir / "lora_adapters.pt"
    if save_lora_adapters_from_checkpoint(args.output_dir, adapter_path):
        run_info["lora_adapters_path"] = str(adapter_path)
        (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        print("Saved LoRA adapters to", adapter_path)
    else:
        print("Could not extract LoRA state from checkpoints; check step_*.tmp layout.")

    # Best checkpoint pointer: fairseq2 may write best to a known path; document or leave as latest
    best_pointer = args.output_dir / "best_checkpoint.txt"
    best_pointer.write_text("Use latest step_* checkpoint or lora_adapters.pt for inference.\n", encoding="utf-8")
    print("Run complete. Config and best pointer in", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
