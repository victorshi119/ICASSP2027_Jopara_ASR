#!/usr/bin/env python3
"""
Phase A: evaluate all MoE branches on v2_test_moe_manifest.jsonl (150 utterances).

Base branches (Z0_base_grn, Z2_base_spa): imported from results/June_09_baseline/.
LoRA branches: model reloaded per branch (deepcopy unsupported by fairseq2), LoRA
injected fresh each time, then inference run on all 150 utterances.

Outputs:
  moe_lora/predictions/all_branches/<branch_id>.jsonl   per-branch predictions
  moe_lora/predictions/branch_matrix.json               WER summary table
  moe_lora/logs/eval_all_branches.log                   progress log

Usage:
  source env.sh && cd $BASE
  $OMNI_PY moe_lora/scripts/eval_all_branches.py
  $OMNI_PY moe_lora/scripts/eval_all_branches.py --branches E1_grn_lora_cegpa,E2_cs_grn_lora
  $OMNI_PY moe_lora/scripts/eval_all_branches.py --force          # re-run all branches
  $OMNI_PY moe_lora/scripts/eval_all_branches.py --batch-size 4   # smaller GPU batches
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

BASE = Path("/media/volume/sapc2_data/victor/icassp2026/Jopara_ASR_models")
MANIFEST = BASE / "moe_lora/manifests/v2_test_moe_manifest.jsonl"
REGISTRY = BASE / "moe_lora/configs/branch_registry.yaml"
BASELINE_DIR = BASE / "results/June_09_baseline"
PRED_DIR = BASE / "moe_lora/predictions/all_branches"
LOG_DIR = BASE / "moe_lora/logs"

log = logging.getLogger(__name__)


def _setup_logging(log_path: Path = LOG_DIR / "eval_all_branches.log") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ── WER / CER (no jiwer dependency) ──────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def _wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else float(len(h))
    return _edit_distance(r, h) / len(r)


def _cer(ref: str, hyp: str) -> float:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    if not r:
        return 0.0 if not h else float(len(h))
    return _edit_distance(r, h) / len(r)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    log.info("Manifest: %d utterances", len(rows))
    return rows


def load_baseline_index() -> dict[str, dict[str, dict]]:
    """Read June_09 baseline CSVs → {lang_mode: {utt_id: csv_row}}"""
    import csv

    result: dict[str, dict[str, dict]] = {}
    for lang in ("grn_Latn", "spa_Latn"):
        lang_dir = BASELINE_DIR / lang
        by_utt: dict[str, dict] = {}
        for csv_path in sorted(lang_dir.glob("test_*.csv")):
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    utt_id = Path(row["audio_path"]).stem
                    by_utt[utt_id] = row
        result[lang] = by_utt
        log.info("Baseline %s: %d utterances loaded", lang, len(by_utt))
    return result


# ── Branch evaluation ─────────────────────────────────────────────────────────

def import_base_branch(
    branch_id: str,
    lang_mode: str,
    manifest_rows: list[dict],
    baseline_index: dict[str, dict[str, dict]],
    out_path: Path,
) -> None:
    by_utt = baseline_index[lang_mode]
    records = []
    missing = []
    for row in manifest_rows:
        uid = row["utt_id"]
        if uid in by_utt:
            b = by_utt[uid]
            records.append({
                "utt_id": uid,
                "branch_id": branch_id,
                "language_mode": lang_mode,
                "subset": row["subset"],
                "v2_category": row["v2_category"],
                "audio_path": row["audio_path"],
                "reference": row["reference_norm"],
                "hypothesis": b["hypothesis"],
                "wer": round(float(b["wer"]), 4),
                "cer": round(float(b["cer"]), 4),
                "source": "imported_june09",
            })
        else:
            missing.append(uid)

    if missing:
        log.warning("Branch %s: %d utt_ids not found in baseline (%s)", branch_id, len(missing), missing[:3])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Branch %s: imported %d rows → %s", branch_id, len(records), out_path.name)


def run_lora_branch(
    branch_id: str,
    branch_cfg: dict,
    manifest_rows: list[dict],
    out_path: Path,
    batch_size: int = 8,
) -> None:
    from fairseq2.models.hub import load_model
    from fairseq2.data.tokenizers.hub import load_tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "lora_adapters"))
    from lora_inject import inject_lora
    from omnilingual_asr.models.inference import ASRInferencePipeline

    adapter_pt = Path(branch_cfg["adapter_path"])
    lora_cfg_path = Path(branch_cfg["lora_config"])
    lora_cfg = json.loads(lora_cfg_path.read_text())
    lang_mode = branch_cfg["language_mode"]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    log.info("Branch %s: loading model (device=%s, dtype=float32) ...", branch_id, device_str)
    t0 = time.time()
    model = load_model("omniASR_LLM_7B_v2", device=torch.device("cpu"), dtype=torch.float32)
    tokenizer = load_tokenizer("omniASR_LLM_7B_v2")
    log.info("  model loaded in %.1fs", time.time() - t0)

    log.info("  injecting LoRA r=%d alpha=%d ...", lora_cfg["r"], lora_cfg["lora_alpha"])
    n = inject_lora(
        model,
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        alpha=lora_cfg["lora_alpha"],
        dropout=lora_cfg["lora_dropout"],
    )
    log.info("  %d LoRA layers injected", n)

    adapter_state = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    missing_keys, _ = model.load_state_dict(adapter_state, strict=False)
    missing_lora = [k for k in missing_keys if "lora_" in k]
    if missing_lora:
        log.error("  missing lora keys: %s", missing_lora[:5])
        sys.exit(1)
    log.info("  adapter loaded: %d tensors from %s", len(adapter_state), adapter_pt.parent.name)

    model.to(device=device, dtype=torch.float32)

    pipeline = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device=device_str,
        dtype=torch.float32,
    )

    # Lift the pipeline's 40s hard cap — it's a soft guard, not an architectural limit.
    # Our manifest has 3 utterances (46s, 46s, 59s) that are valid and must be transcribed.
    import omnilingual_asr.models.inference.pipeline as _pipe
    _pipe.MAX_ALLOWED_AUDIO_SEC = float("inf")

    audio_paths = [r["audio_path"] for r in manifest_rows]
    lang_tags = [lang_mode] * len(audio_paths)

    log.info("  running inference on %d utterances (batch_size=%d, lang=%s) ...",
             len(audio_paths), batch_size, lang_mode)
    t0 = time.time()
    hyp_map: dict[str, str] = {}
    for i in range(0, len(audio_paths), batch_size):
        batch_audio = audio_paths[i:i + batch_size]
        batch_lang = lang_tags[i:i + batch_size]
        hyps = pipeline.transcribe(batch_audio, lang=batch_lang)
        hyps = hyps if hyps else [""] * len(batch_audio)
        for path, hyp in zip(batch_audio, hyps):
            utt_id = Path(path).stem
            hyp_map[utt_id] = hyp if isinstance(hyp, str) else str(hyp)
        done = min(i + batch_size, len(audio_paths))
        if done % 40 == 0 or done == len(audio_paths):
            elapsed = time.time() - t0
            log.info("    %d/%d utterances done (%.0fs elapsed, %.2fs/utt)",
                     done, len(audio_paths), elapsed, elapsed / done)

    log.info("  inference complete in %.1fs", time.time() - t0)

    records = []
    for row in manifest_rows:
        ref = row["reference_norm"]
        uid = row["utt_id"]
        hyp_str = hyp_map.get(uid, "")
        records.append({
            "utt_id": uid,
            "branch_id": branch_id,
            "language_mode": lang_mode,
            "subset": row["subset"],
            "v2_category": row["v2_category"],
            "audio_path": row["audio_path"],
            "reference": ref,
            "hypothesis": hyp_str,
            "wer": round(_wer(ref, hyp_str), 4),
            "cer": round(_cer(ref, hyp_str), 4),
            "source": "inference",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Branch %s: wrote %d rows → %s", branch_id, len(records), out_path.name)

    del model, pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Branch matrix ─────────────────────────────────────────────────────────────

SUBSET_KEYS = ["test_guarani_only", "test_mixed", "test_jehea", "test_spanish_only"]
SUBSET_SHORT = ["grn_only", "mixed", "jehea", "spa_only"]


def build_branch_matrix(all_branch_ids: list[str], pred_dir: Path = PRED_DIR) -> dict:
    matrix = {}
    for bid in all_branch_ids:
        path = pred_dir / f"{bid}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        by_subset: dict[str, list[float]] = {}
        for r in rows:
            sub = r["subset"]
            if r["wer"] is not None:
                by_subset.setdefault(sub, []).append(float(r["wer"]))

        branch_stats: dict = {"overall": None, "by_subset": {}}
        all_wers: list[float] = []
        for sk in SUBSET_KEYS:
            wers = by_subset.get(sk, [])
            if wers:
                avg = round(sum(wers) / len(wers), 4)
                branch_stats["by_subset"][sk] = {"n": len(wers), "wer": avg}
                all_wers.extend(wers)
            else:
                branch_stats["by_subset"][sk] = {"n": 0, "wer": None}
        if all_wers:
            branch_stats["overall"] = round(sum(all_wers) / len(all_wers), 4)
        matrix[bid] = branch_stats
    return matrix


def print_matrix(matrix: dict) -> None:
    col_w = 12
    header = f"{'Branch':<28}" + "".join(f"{s:>{col_w}}" for s in SUBSET_SHORT + ["overall"])
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for bid, stats in matrix.items():
        vals = []
        for sk in SUBSET_KEYS:
            v = stats["by_subset"].get(sk, {}).get("wer")
            vals.append(f"{v:.3f}" if v is not None else "  N/A")
        ov = stats["overall"]
        vals.append(f"{ov:.3f}" if ov is not None else "  N/A")
        print(f"{bid:<28}" + "".join(f"{v:>{col_w}}" for v in vals))
    print("═" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches", help="Comma-separated branch IDs to run (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-run even if output JSONL exists")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size for LoRA branches")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="Path to manifest JSONL (default: v2_test)")
    parser.add_argument("--out-dir", type=Path, default=PRED_DIR, help="Output directory for branch JSONLs")
    args = parser.parse_args()

    # Allow overriding manifest and output dir (used for dev set eval)
    manifest_path = args.manifest
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"eval_{out_dir.name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_path)

    log.info("═" * 60)
    log.info("MoE branch evaluation — JetStream2")
    log.info("  manifest: %s", manifest_path)
    log.info("  out-dir:  %s", out_dir)
    log.info("═" * 60)

    manifest_rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    log.info("Manifest: %d utterances", len(manifest_rows))
    baseline_index = load_baseline_index()

    with REGISTRY.open() as f:
        registry = yaml.safe_load(f)
    branches: dict = registry["branches"]

    selected = args.branches.split(",") if args.branches else list(branches.keys())
    log.info("Branches selected: %s", selected)

    t_start = time.time()
    for branch_id in selected:
        if branch_id not in branches:
            log.error("Unknown branch '%s'. Registry has: %s", branch_id, list(branches.keys()))
            continue

        out_path = out_dir / f"{branch_id}.jsonl"
        if out_path.exists() and not args.force:
            n = sum(1 for l in out_path.read_text().splitlines() if l.strip())
            log.info("Branch %s: output exists (%d rows), skipping (--force to rerun)", branch_id, n)
            continue

        cfg = branches[branch_id]
        log.info("\n── Branch %s [%s, %s] ─────────────────────────", branch_id, cfg["kind"], cfg["language_mode"])

        if cfg["kind"] == "base":
            import_base_branch(branch_id, cfg["language_mode"], manifest_rows, baseline_index, out_path)
        elif cfg["kind"] == "lora":
            run_lora_branch(branch_id, cfg, manifest_rows, out_path, batch_size=args.batch_size)
        else:
            log.error("Branch %s: unrecognized kind '%s'", branch_id, cfg["kind"])

    log.info("\n── Building branch matrix ────────────────────────────────────")
    matrix = build_branch_matrix(list(branches.keys()), out_dir)

    matrix_path = out_dir.parent / f"branch_matrix_{out_dir.name}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    log.info("Branch matrix → %s", matrix_path)

    print_matrix(matrix)

    elapsed = time.time() - t_start
    log.info("Total wall time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
