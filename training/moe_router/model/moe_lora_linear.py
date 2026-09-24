#!/usr/bin/env python3
"""
MoELoRALinear: one Linear layer with 3 LoRA expert deltas soft-mixed by utterance gate values.

Forward formula (per adapted decoder layer):
    y = W0(x)
      + p_spa * scale_spa * B_spa(A_spa(x))
      + p_cs  * scale_cs  * B_cs (A_cs (x))
      + p_grn * scale_grn * B_grn(A_grn(x))

where scale_k = alpha_k / rank_k.

Gate values [p_spa, p_cs, p_grn] come from a shared GateStore that the
SoftRegimeRouter writes to at the start of each decoder forward pass.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

EXPERT_KEYS = ("spa", "cs", "grn")


# ---------------------------------------------------------------------------
# Shared gate storage
# ---------------------------------------------------------------------------

class GateStore:
    """
    Thread-local storage for the current utterance-regime gate vector.
    The SoftRegimeRouter writes here; all MoELoRALinear instances read from here.
    One GateStore instance is shared by all MoELoRALinear layers in a patched model.
    """

    def __init__(self) -> None:
        self._gates: torch.Tensor | None = None
        self._locked: bool = False

    def set(self, gates: torch.Tensor) -> None:
        # gates: [3] or [B, 3] float tensor on any device
        # No-op if locked — preserves a manually set one-hot gate against the encoder hook
        if not self._locked:
            self._gates = gates

    def set_locked(self, gates: torch.Tensor) -> None:
        """Set gates and lock so the encoder forward hook cannot overwrite them."""
        self._gates = gates
        self._locked = True

    def unlock(self) -> None:
        self._locked = False

    def get(self) -> torch.Tensor | None:
        return self._gates

    def clear(self) -> None:
        self._gates = None
        self._locked = False


# ---------------------------------------------------------------------------
# Helpers (mirror of lora_inject._linear_in_out without the import dependency)
# ---------------------------------------------------------------------------

def _linear_in_out(linear: nn.Module) -> tuple[int, int]:
    if hasattr(linear, "in_features") and hasattr(linear, "out_features"):
        return int(linear.in_features), int(linear.out_features)
    if hasattr(linear, "input_dim") and hasattr(linear, "output_dim"):
        return int(linear.input_dim), int(linear.output_dim)
    raise AttributeError(f"Cannot infer dims from {type(linear)}: missing in_features/input_dim")


# ---------------------------------------------------------------------------
# MoELoRALinear
# ---------------------------------------------------------------------------

class MoELoRALinear(nn.Module):
    """
    Drop-in replacement for a plain Linear (or LinearWithLoRA) inside the
    OmniASR decoder.  Stores three LoRA expert pairs (A_k, B_k) and mixes
    their deltas using soft gates from a shared GateStore.

    Parameters
    ----------
    linear       : the frozen base Linear module (torch.nn.Linear or fairseq2 Linear)
    gate_store   : GateStore shared with the SoftRegimeRouter
    experts      : dict with keys "spa", "cs", "grn", each value a dict:
                     {"A": Tensor[r, in], "B": Tensor[out, r], "r": int, "alpha": float}
    dropout      : dropout applied to the LoRA intermediate activations
    """

    def __init__(
        self,
        linear: nn.Module,
        gate_store: GateStore,
        experts: dict[str, dict[str, Any]],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.gate_store = gate_store

        in_dim, out_dim = _linear_in_out(linear)

        # Build A/B projections and scaling for each expert
        self.A: nn.ModuleDict = nn.ModuleDict()
        self.B: nn.ModuleDict = nn.ModuleDict()
        self.scales: dict[str, float] = {}

        for key in EXPERT_KEYS:
            cfg = experts[key]
            r     = int(cfg["r"])
            alpha = float(cfg["alpha"])
            A_w   = cfg["A"]   # [r, in_dim]
            B_w   = cfg["B"]   # [out_dim, r]

            assert A_w.shape == (r, in_dim),  f"Expert {key} A shape mismatch: {A_w.shape} vs ({r}, {in_dim})"
            assert B_w.shape == (out_dim, r), f"Expert {key} B shape mismatch: {B_w.shape} vs ({out_dim}, {r})"

            A_proj = nn.Linear(in_dim, r, bias=False)
            B_proj = nn.Linear(r, out_dim, bias=False)
            A_proj.weight.data.copy_(A_w)
            B_proj.weight.data.copy_(B_w)

            self.A[key] = A_proj
            self.B[key] = B_proj
            self.scales[key] = alpha / r

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers used by the training schedule
    # ------------------------------------------------------------------

    def freeze_experts(self) -> None:
        for key in EXPERT_KEYS:
            for p in self.A[key].parameters():
                p.requires_grad = False
            for p in self.B[key].parameters():
                p.requires_grad = False

    def unfreeze_experts(self) -> None:
        for key in EXPERT_KEYS:
            for p in self.A[key].parameters():
                p.requires_grad = True
            for p in self.B[key].parameters():
                p.requires_grad = True

    def freeze_base(self) -> None:
        for p in self.linear.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)

        gates = self.gate_store.get()
        if gates is not None:
            gates = gates.to(dtype=x.dtype, device=x.device)
        if gates is None:
            # No gates set — return uniform mixture (should not happen after router wires up)
            deltas = [
                self.scales[k] * self.B[k](self.dropout(self.A[k](x)))
                for k in EXPERT_KEYS
            ]
            return base + sum(deltas) / len(EXPERT_KEYS)

        # gates: [3] (single utt) or [B, 3] (batch)
        # x:     [T, D] or [B, T, D]  (or any prefix shape followed by D)
        # We broadcast gates across all non-expert dims via unsqueezing.
        p_list = [gates[..., i : i + 1] for i in range(len(EXPERT_KEYS))]
        # p_list[k]: [..., 1]; add dims to match x
        while p_list[0].dim() < x.dim():
            p_list = [p.unsqueeze(-1) for p in p_list]

        out = base
        for k, p_k in zip(EXPERT_KEYS, p_list):
            delta_k = self.scales[k] * self.B[k](self.dropout(self.A[k](x)))
            out = out + p_k * delta_k

        return out

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_expert_state_dict(self) -> dict[str, torch.Tensor]:
        """Return all expert A/B weights keyed by 'A_<key>.weight' / 'B_<key>.weight'."""
        sd: dict[str, torch.Tensor] = {}
        for key in EXPERT_KEYS:
            sd[f"A_{key}.weight"] = self.A[key].weight.detach().cpu()
            sd[f"B_{key}.weight"] = self.B[key].weight.detach().cpu()
        return sd

    def load_expert_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Restore expert A/B weights from a dict produced by get_expert_state_dict()."""
        for key in EXPERT_KEYS:
            a_key = f"A_{key}.weight"
            b_key = f"B_{key}.weight"
            if a_key in sd:
                self.A[key].weight.data.copy_(sd[a_key])
            if b_key in sd:
                self.B[key].weight.data.copy_(sd[b_key])


# ---------------------------------------------------------------------------
# Factory: build MoELoRALinear from raw adapter state dicts
# ---------------------------------------------------------------------------

def load_expert_ab(
    adapter_path: str,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load A and B weight matrices for a specific decoder module from a saved adapter.

    The adapter state dict uses keys like:
        '<module_name>.lora_A.weight'  → A matrix [r, in_dim]
        '<module_name>.lora_B.weight'  → B matrix [out_dim, r]

    Returns (A, B) or None if the module is not in the adapter.
    """
    import torch
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    key_A = f"{module_name}.lora_A.weight"
    key_B = f"{module_name}.lora_B.weight"
    if key_A not in state or key_B not in state:
        return None
    return state[key_A], state[key_B]


def build_moe_lora_linear(
    linear: nn.Module,
    gate_store: GateStore,
    module_name: str,
    expert_cfgs: list[dict],
    dropout: float = 0.0,
) -> "MoELoRALinear | None":
    """
    Construct a MoELoRALinear for the given module from three expert adapter files.

    expert_cfgs: list of 3 dicts (spa, cs, grn order), each with:
        adapter_path : str
        r            : int
        alpha        : float

    Returns None if any expert is missing weights for this module.
    """
    expert_dicts: dict[str, dict] = {}
    for key, cfg in zip(EXPERT_KEYS, expert_cfgs):
        ab = load_expert_ab(cfg["adapter_path"], module_name)
        if ab is None:
            return None
        A, B = ab
        expert_dicts[key] = {"A": A, "B": B, "r": cfg["r"], "alpha": cfg["alpha"]}
    return MoELoRALinear(linear, gate_store, expert_dicts, dropout=dropout)
