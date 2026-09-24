#!/usr/bin/env python3
"""
LoRA injection for OmniASR LLM (fairseq2 Wav2Vec2LlamaModel).
Wraps decoder Linear layers (q_proj, v_proj, etc.) with low-rank adapters;
freeze_non_lora() keeps only LoRA parameters trainable.
Supports both torch.nn.Linear and fairseq2.nn.Linear (Projection; uses input_dim/output_dim).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _linear_in_out(linear: nn.Module) -> tuple[int, int]:
    """Return (in_features, out_features) for torch.nn.Linear or fairseq2.nn.Linear."""
    if hasattr(linear, "in_features") and hasattr(linear, "out_features"):
        return int(linear.in_features), int(linear.out_features)
    if hasattr(linear, "input_dim") and hasattr(linear, "output_dim"):
        return int(linear.input_dim), int(linear.output_dim)
    raise AttributeError(f"Module {type(linear)} has no in_features/out_features or input_dim/output_dim")


def _is_linear_like(mod: nn.Module) -> bool:
    """True if mod is torch.nn.Linear or fairseq2-style Linear (Projection with weight)."""
    if isinstance(mod, nn.Linear):
        return True
    return (
        hasattr(mod, "weight")
        and hasattr(mod, "forward")
        and (getattr(mod, "input_dim", None) is not None or getattr(mod, "in_features", None) is not None)
    )


class LinearWithLoRA(nn.Module):
    """Wraps a Linear (torch or fairseq2) with LoRA: out = linear(x) + (alpha/r) * lora_B(dropout(lora_A(x)))."""

    def __init__(
        self,
        linear: nn.Module,
        r: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = linear
        in_dim, out_dim = _linear_in_out(linear)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r else 0.0
        self.lora_dropout = nn.Dropout(p=dropout)
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return base + self.scaling * lora_out


def _parent_and_attr(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    """Return (parent_module, attribute_name) for the given full parameter name."""
    parts = full_name.split(".")
    parent = model
    for i in range(len(parts) - 1):
        parent = getattr(parent, parts[i])
    return parent, parts[-1]


def _is_decoder_linear_name(name: str) -> bool:
    """True if this module path looks like the LLM/decoder (not encoder frontend/encoder)."""
    name_lower = name.lower()
    # Exclude wav2vec2 encoder
    if "encoder_frontend" in name_lower or (name_lower.startswith("encoder.") and "decoder" not in name_lower):
        return False
    if "encoder." in name_lower and "encoder_proj" not in name_lower and "decoder" not in name_lower:
        return False
    # Include known decoder paths (omnilingual-asr uses llama_decoder; some checkpoints may use decoder/layers)
    if "llama_decoder" in name_lower or "decoder" in name_lower:
        return True
    # Transformer decoder layers (e.g. layers.0.self_attn) but not encoder layers
    if ".layers." in name_lower and "encoder" not in name_lower:
        return True
    return False


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> int:
    """
    Replace decoder Linear layers whose name matches any of target_modules with LinearWithLoRA.
    target_modules are matched as substrings of the full module name (e.g. "q_proj", "v_proj").
    Decoder path: llama_decoder, decoder, or .layers. (excluding encoder).
    Returns the number of layers replaced.
    """
    replaced = 0
    to_replace: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not _is_linear_like(mod):
            continue
        if not _is_decoder_linear_name(name):
            continue
        segments = name.split(".")
        if any(seg in target_modules for seg in segments):
            parent, attr = _parent_and_attr(model, name)
            to_replace.append((parent, attr, mod))
    for parent, attr, linear in to_replace:
        wrapped = LinearWithLoRA(linear, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        replaced += 1
    return replaced


def _list_decoder_linears(model: nn.Module) -> None:
    """Print full names of Linear-like layers (for debugging 0 injection)."""
    decoder_like = [
        name for name, mod in model.named_modules()
        if _is_linear_like(mod) and ("decoder" in name.lower() or "llama" in name.lower())
    ]
    all_linears = [name for name, mod in model.named_modules() if _is_linear_like(mod)]
    print(f"[LoRA] Decoder/llama Linear names ({len(decoder_like)}): {decoder_like[:20]}{'...' if len(decoder_like) > 20 else ''}", flush=True)
    print(f"[LoRA] ALL Linear names (first 40): {all_linears[:40]}", flush=True)


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze all parameters, then unfreeze those with 'lora_' in the name."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Return number of parameters (optionally only trainable)."""
    return sum(
        p.numel()
        for p in model.parameters()
        if (not only_trainable or p.requires_grad)
    )


def get_lora_state_dict(model: nn.Module) -> dict[str, Any]:
    """Return state_dict containing only LoRA parameters (for saving adapters)."""
    return {
        name: param.cpu().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }
