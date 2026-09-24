#!/usr/bin/env python3
"""
SoftRegimeRouter: MLP on pooled OmniASR encoder states → [p_spa, p_cs, p_grn].

The router sits between the encoder and the MoELoRALinear layers.
It pools the encoder's frame-level output into a single vector and predicts
a soft distribution over the three utterance regimes:
    spa_only, cs_jopara, grn_only

After forward(), the gate vector is written to a shared GateStore so that all
MoELoRALinear modules in the decoder read the same gates during that pass.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolingMode = Literal["mean", "mean_std", "attention"]

REGIME_ORDER = ["spa", "cs", "grn"]   # index 0 = spa, 1 = cs, 2 = grn


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learnable single-head attention pooling: h = sum_t(alpha_t * h_t)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # enc_out: [T, D] or [B, T, D]
        scores = self.proj(enc_out).squeeze(-1)       # [T] or [B, T]
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        alpha  = torch.softmax(scores, dim=-2 if enc_out.dim() == 3 else -1)  # [T] or [B, T]
        if enc_out.dim() == 3:
            return (alpha.unsqueeze(-1) * enc_out).sum(dim=1)   # [B, D]
        return (alpha.unsqueeze(-1) * enc_out).sum(dim=0)       # [D]


def pool_encoder_output(
    enc_out: torch.Tensor,
    mode: PoolingMode,
    attn_pool: AttentionPooling | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pool [T, D] or [B, T, D] encoder output to [D] or [B, D].

    Modes:
      mean       → mean over T
      mean_std   → concat(mean, std) → [2D]
      attention  → learnable attention pooling → [D]
    """
    if mode == "attention":
        assert attn_pool is not None
        return attn_pool(enc_out, padding_mask)

    if enc_out.dim() == 2:   # [T, D]
        mean = enc_out.mean(dim=0)
        if mode == "mean_std":
            std = enc_out.std(dim=0)
            return torch.cat([mean, std], dim=-1)
        return mean
    else:                     # [B, T, D]
        mean = enc_out.mean(dim=1)
        if mode == "mean_std":
            std = enc_out.std(dim=1)
            return torch.cat([mean, std], dim=-1)
        return mean


# ---------------------------------------------------------------------------
# Router MLP
# ---------------------------------------------------------------------------

class SoftRegimeRouter(nn.Module):
    """
    Utterance-level soft regime router.

    Input : pooled OmniASR encoder states  → [D] or [B, D]
    Output: soft regime posterior           → [3] or [B, 3]  (sum to 1)

    The output is also written to a GateStore that MoELoRALinear reads during
    the decoder forward pass.
    """

    def __init__(
        self,
        d_enc: int = 4096,
        hidden: int = 512,
        n_classes: int = 3,
        pooling: PoolingMode = "mean_std",
        dropout: float = 0.1,
        gate_store=None,       # GateStore | None — if given, forward() writes gates
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.gate_store = gate_store
        self.n_classes = n_classes

        d_in = d_enc * 2 if pooling == "mean_std" else d_enc
        self.attn_pool = AttentionPooling(d_enc) if pooling == "attention" else None

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        enc_out: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_out      : encoder hidden states  [T, D] or [B, T, D]
        padding_mask : bool mask, True = padding  [T] or [B, T]

        Returns
        -------
        gates : softmax-normalized regime probabilities  [3] or [B, 3]
        """
        pooled = pool_encoder_output(enc_out, self.pooling, self.attn_pool, padding_mask)
        logits = self.mlp(pooled)               # [3] or [B, 3]
        gates  = torch.softmax(logits, dim=-1)

        if self.gate_store is not None:
            self.gate_store.set(gates.detach() if not gates.requires_grad else gates)

        return gates

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def kl_loss(
        self,
        gates: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(soft_targets ‖ gates)  — push predicted distribution toward the soft labels.

        gates        : [3] or [B, 3]  (model predictions, post-softmax)
        soft_targets : [3] or [B, 3]  (from build_soft_regime_targets.py)
        """
        log_gates = torch.log(gates.clamp(min=1e-8))
        return F.kl_div(log_gates, soft_targets, reduction="batchmean")

    def entropy_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        H(gates) = -sum(p * log p)  (mean over batch).
        Adding this to the total loss encourages diverse routing (prevents collapse).
        """
        log_gates = torch.log(gates.clamp(min=1e-8))
        return -(gates * log_gates).sum(dim=-1).mean()

    def combined_regime_loss(
        self,
        gates: torch.Tensor,
        soft_targets: torch.Tensor,
        lambda_kl: float = 0.2,
        lambda_ent: float = 0.001,
    ) -> torch.Tensor:
        """Convenience: lambda_kl * KL + lambda_ent * H(p)."""
        return lambda_kl * self.kl_loss(gates, soft_targets) + \
               lambda_ent * self.entropy_loss(gates)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "pooling": self.pooling}, path)
        print(f"[Router] saved → {path}")

    @classmethod
    def load(cls, path: str, gate_store=None, **kwargs) -> "SoftRegimeRouter":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        pooling = ckpt.get("pooling", "mean_std")
        router = cls(pooling=pooling, gate_store=gate_store, **kwargs)
        router.load_state_dict(ckpt["state_dict"])
        print(f"[Router] loaded ← {path}")
        return router


# ---------------------------------------------------------------------------
# Soft target construction (used by build_soft_regime_targets.py and training)
# ---------------------------------------------------------------------------

def make_soft_target(
    regime: str,
    spa_count: int = 0,
    grn_count: int = 0,
    jha_count: int = 0,
) -> list[float]:
    """
    Build a soft [p_spa, p_cs, p_grn] target for one utterance.

    For spa/grn regimes, use fixed soft labels that leave small residual mass
    on other classes.  For CS-Jopara, distribute the 30% residual according to
    word-level evidence so the router can learn dominance direction.
    """
    if regime == "spa":
        return [0.90, 0.05, 0.05]
    if regime == "grn":
        return [0.05, 0.05, 0.90]

    # CS-Jopara: evidence-weighted residual
    spa_ev = float(spa_count) + 0.5 * float(jha_count)
    grn_ev = float(grn_count) + 0.5 * float(jha_count)
    total  = spa_ev + grn_ev
    if total < 1.0:
        return [0.10, 0.80, 0.10]

    q_cs  = 0.70
    q_spa = 0.30 * spa_ev / total
    q_grn = 0.30 * grn_ev / total
    return [round(q_spa, 4), round(q_cs, 4), round(q_grn, 4)]


def soft_target_tensor(
    regimes: list[str],
    spa_counts: list[int] | None = None,
    grn_counts: list[int] | None = None,
    jha_counts: list[int] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Batch version: returns [B, 3] float tensor."""
    n = len(regimes)
    spa_counts = spa_counts or [0] * n
    grn_counts = grn_counts or [0] * n
    jha_counts = jha_counts or [0] * n
    targets = [
        make_soft_target(r, s, g, j)
        for r, s, g, j in zip(regimes, spa_counts, grn_counts, jha_counts)
    ]
    return torch.tensor(targets, dtype=torch.float32, device=device)
