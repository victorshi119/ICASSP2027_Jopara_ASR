#!/usr/bin/env python3
"""
Stage A2 — Train the SoftRegimeRouter using pre-extracted OmniASR encoder embeddings.

This is the fast path: encoder embeddings are already on disk from June_14.
We train only the router MLP (no OmniASR forward pass needed).
Loss: lambda_kl * KL(soft_target ‖ p_router) + lambda_entropy * H(p_router)

Outputs (in checkpoints/jmole_grn_router/ or as specified in config):
  router_best.pt        best router checkpoint (by dev KL loss)
  router_final.pt       final checkpoint
  router_train_log.jsonl  per-epoch metrics
  router_dev_results.json  per-utterance dev predictions

Usage:
  $OMNI_PY scripts/train_router_on_embeddings.py --config configs/jmole_grn.yaml
  $OMNI_PY scripts/train_router_on_embeddings.py --config configs/jmole_grn.yaml --epochs 100
  $OMNI_PY scripts/train_router_on_embeddings.py --config configs/jmole_grn.yaml --seed 42
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
from pathlib import Path

_PROJ_ROOT = Path(os.environ.get("JOPARA_ROOT", "/N/project/icassp2026/Jopara_ASR_models_js2"))  # data root (original project layout)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "lora_adapters"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from model.soft_regime_router import (
    SoftRegimeRouter,
    REGIME_ORDER,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings_and_targets(
    emb_path: str,
    targets_path: str,
    utt_ids_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Load pre-extracted encoder embeddings and match them to soft targets.

    emb_path     : .npy  [N, D]
    targets_path : .jsonl  {utt_id, soft_target, regime, ...}
    utt_ids_path : .json  list of utt_ids in the same row order as emb_path
    """
    embs = np.load(emb_path)                                 # [N, D]
    utt_ids: list[str]

    # Load utterance ID order that matches the embedding rows
    if utt_ids_path and Path(utt_ids_path).exists():
        utt_ids = json.loads(Path(utt_ids_path).read_text())
    else:
        # Derive utt_ids_path from emb_path convention: *_embeddings.npy → *_utt_ids.json
        ids_path = emb_path.replace("_embeddings.npy", "_utt_ids.json")
        utt_ids = json.loads(Path(ids_path).read_text())

    # Load soft targets; build two lookup dicts:
    #   target_map      keyed by utt_id    (session-level: "tr-010", "CV")
    #   target_map_stem keyed by audio_stem (segment-level: "tr-010_02586")
    # Train embeddings use session IDs; dev CEGPA embeddings use segment IDs.
    # We try utt_id first, then audio_stem, so both match correctly.
    target_rows = [
        json.loads(l) for l in Path(targets_path).read_text().splitlines() if l.strip()
    ]
    target_map: dict[str, list[float]] = {r["utt_id"]: r["soft_target"] for r in target_rows}
    target_map_stem: dict[str, list[float]] = {
        r["audio_stem"]: r["soft_target"] for r in target_rows if "audio_stem" in r
    }

    # Align embeddings to soft targets (keep only rows that have both)
    kept_embs: list[np.ndarray] = []
    kept_targets: list[list[float]] = []
    kept_ids: list[str] = []
    missing = 0
    for uid, emb in zip(utt_ids, embs):
        if uid in target_map:
            tgt = target_map[uid]
        elif uid in target_map_stem:
            tgt = target_map_stem[uid]
        else:
            missing += 1
            continue
        kept_embs.append(emb)
        kept_targets.append(tgt)
        kept_ids.append(uid)

    if missing:
        print(f"  WARNING: {missing} embeddings have no soft target — dropped")

    X = torch.tensor(np.stack(kept_embs), dtype=torch.float32)         # [N, D]
    y = torch.tensor(kept_targets, dtype=torch.float32)                 # [N, 3]

    print(f"  Loaded: {X.shape[0]} utterances, embedding dim={X.shape[1]}")
    return X, y, kept_ids


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(router: SoftRegimeRouter, loader: DataLoader, device: torch.device,
             lambda_kl: float, lambda_ent: float) -> dict[str, float]:
    router.eval()
    total_kl = total_ent = total_acc = n_batches = n_utt = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            # Route through MLP directly (embeddings already pooled)
            logits = router.mlp(X_b)                      # [B, 3]
            probs  = torch.softmax(logits, dim=-1)

            kl  = F.kl_div(torch.log(probs.clamp(1e-8)), y_b, reduction="batchmean")
            ent = -(probs * torch.log(probs.clamp(1e-8))).sum(-1).mean()

            pred_regime = probs.argmax(-1)
            gold_regime = y_b.argmax(-1)
            acc = (pred_regime == gold_regime).float().mean()

            total_kl  += kl.item()
            total_ent += ent.item()
            total_acc += acc.item()
            n_batches += 1
            n_utt     += X_b.shape[0]

    n = max(n_batches, 1)
    return {
        "kl":   total_kl  / n,
        "ent":  total_ent / n,
        "acc":  total_acc / n,
        "loss": lambda_kl * (total_kl / n) + lambda_ent * (total_ent / n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jmole_grn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config max_epochs")
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--batch",  type=int,   default=None)
    parser.add_argument("--seed",   type=int,   default=None,
                         help="Random seed for router init + data-loader shuffling (default: unseeded/nondeterministic)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                         help="Override config's training.a2.checkpoint_dir (e.g. for per-seed output dirs)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parents[2] / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    a2 = cfg["training"]["a2"]
    router_cfg = cfg.get("router", {})

    # Overrides from CLI
    max_epochs  = args.epochs or a2.get("max_epochs", 50)
    lr          = args.lr or float(a2.get("lr", 1e-3))
    batch_size  = args.batch or int(a2.get("batch_size", 128))
    lambda_kl   = float(a2.get("lambda_kl", 0.2))
    lambda_ent  = float(a2.get("lambda_entropy", 0.001))
    patience    = int(a2.get("patience", 10))
    ckpt_dir    = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(a2["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading train embeddings + soft targets …")
    X_tr, y_tr, ids_tr = load_embeddings_and_targets(
        a2["embeddings_train"],
        a2["soft_targets_train"],
    )
    print("Loading dev embeddings + soft targets …")
    X_dv, y_dv, ids_dv = load_embeddings_and_targets(
        a2["embeddings_dev"],
        a2["soft_targets_dev"],
    )

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,  drop_last=False)
    dev_loader   = DataLoader(TensorDataset(X_dv, y_dv), batch_size=batch_size, shuffle=False, drop_last=False)

    # ── Build router ──────────────────────────────────────────────────────────
    d_enc = X_tr.shape[1]
    # If pooling = "mean_std", d_in = 2*D but embeddings are already pooled with mean
    # (June_14 extract_encoder_embeddings used mean pooling → [D] not [2D])
    # Force "mean" pooling here since embeddings are already pooled [D] vectors.
    pooling = "mean"
    print(f"\nNote: pre-extracted embeddings are already mean-pooled (D={d_enc}). "
          f"Using pooling='mean' regardless of config. For attention/mean_std pooling, "
          f"re-extract embeddings with that pooling mode.")

    router = SoftRegimeRouter(
        d_enc   = d_enc,
        hidden  = router_cfg.get("hidden", 512),
        pooling = pooling,
        dropout = router_cfg.get("dropout", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in router.parameters())
    print(f"Router parameters: {n_params:,}")

    optim = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dev_loss = float("inf")
    best_epoch    = -1
    no_improve    = 0
    log_records: list[dict] = []

    print(f"\nTraining router for up to {max_epochs} epochs …")
    print(f"  lr={lr}, batch={batch_size}, lambda_kl={lambda_kl}, lambda_ent={lambda_ent}")

    for epoch in range(1, max_epochs + 1):
        router.train()
        total_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = router.mlp(X_b)
            probs  = torch.softmax(logits, dim=-1)

            kl_loss  = F.kl_div(torch.log(probs.clamp(1e-8)), y_b, reduction="batchmean")
            ent_loss = -(probs * torch.log(probs.clamp(1e-8))).sum(-1).mean()
            loss = lambda_kl * kl_loss + lambda_ent * ent_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        dev_metrics = evaluate(router, dev_loader, device, lambda_kl, lambda_ent)

        rec = {"epoch": epoch, "train_loss": round(train_loss, 5), **{f"dev_{k}": round(v, 5) for k, v in dev_metrics.items()}}
        log_records.append(rec)

        improved = dev_metrics["loss"] < best_dev_loss
        if improved:
            best_dev_loss = dev_metrics["loss"]
            best_epoch    = epoch
            no_improve    = 0
            router.save(str(ckpt_dir / "router_best.pt"))
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | "
                  f"dev_kl={dev_metrics['kl']:.4f} dev_acc={dev_metrics['acc']:.3f} "
                  f"{'← best' if improved else ''}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Save final checkpoint and logs ────────────────────────────────────────
    router.save(str(ckpt_dir / "router_final.pt"))
    (ckpt_dir / "router_train_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in log_records)
    )
    print(f"\nBest dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")

    # ── Per-utterance dev predictions ─────────────────────────────────────────
    print("\nGenerating dev predictions …")
    router_best = SoftRegimeRouter.load(str(ckpt_dir / "router_best.pt")).to(device)
    router_best.eval()
    results: list[dict] = []
    with torch.no_grad():
        for i, (uid, emb, tgt) in enumerate(zip(ids_dv, X_dv, y_dv)):
            probs = torch.softmax(router_best.mlp(emb.unsqueeze(0).to(device)), dim=-1)[0]
            pred_idx = probs.argmax().item()
            gold_idx = tgt.argmax().item()
            results.append({
                "utt_id":      uid,
                "pred_regime": REGIME_ORDER[pred_idx],
                "gold_regime": REGIME_ORDER[gold_idx],
                "correct":     pred_idx == gold_idx,
                "p_spa":  round(probs[0].item(), 4),
                "p_cs":   round(probs[1].item(), 4),
                "p_grn":  round(probs[2].item(), 4),
                "soft_target": [round(v, 4) for v in tgt.tolist()],
            })

    (ckpt_dir / "router_dev_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    acc  = sum(r["correct"] for r in results) / len(results)
    print(f"Dev accuracy: {acc:.3f}  ({sum(r['correct'] for r in results)}/{len(results)})")

    # Per-regime accuracy
    for reg in REGIME_ORDER:
        subset = [r for r in results if r["gold_regime"] == reg]
        if subset:
            reg_acc = sum(r["correct"] for r in subset) / len(subset)
            print(f"  {reg}: {reg_acc:.3f}  (n={len(subset)})")

    print(f"\nAll outputs → {ckpt_dir}")


if __name__ == "__main__":
    main()
