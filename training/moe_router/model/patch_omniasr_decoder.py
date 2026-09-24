#!/usr/bin/env python3
"""
Patch an OmniASR (Wav2Vec2LlamaModel) decoder with MoELoRALinear and attach a
SoftRegimeRouter forward hook so the model runs as a unified J-MoLE-ASR model.

Usage (see train_jmole_asr.py for the full training script):

    from patch_omniasr_decoder import build_jmole_model
    model, router, gate_store = build_jmole_model(config)

After this call:
    - Every q_proj / v_proj in the decoder is a MoELoRALinear.
    - A forward hook on the encoder writes router gates to gate_store before
      the decoder attention layers run.
    - model.encoder and the base LLM backbone are frozen by default.
    - The router MLP and (optionally) the LoRA experts are trainable.
"""
from __future__ import annotations

import json
import sys
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Make sibling model imports available
_MODEL_DIR = Path(__file__).parent
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent)); sys.path.insert(0, str(_HERE.parents[1] / "lora_adapters"))

from model.moe_lora_linear import (
    GateStore,
    MoELoRALinear,
    EXPERT_KEYS,
    build_moe_lora_linear,
    _linear_in_out,
)
from model.soft_regime_router import (
    SoftRegimeRouter,
)

# Reuse existing decoder-name helpers from lora_inject.py
from lora_inject import (
    _is_decoder_linear_name,
    _is_linear_like,
    _parent_and_attr,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level: scan model for patchable layers
# ---------------------------------------------------------------------------

def _find_target_linears(
    model: nn.Module,
    target_modules: list[str],
) -> list[tuple[str, nn.Module, nn.Module, str]]:
    """
    Return (full_name, parent_module, child_module, attr_name) for each
    decoder Linear that matches any of target_modules.
    """
    found: list[tuple[str, nn.Module, nn.Module, str]] = []
    for name, mod in model.named_modules():
        if not _is_linear_like(mod):
            continue
        if not _is_decoder_linear_name(name):
            continue
        segments = name.split(".")
        if any(seg in target_modules for seg in segments):
            parent, attr = _parent_and_attr(model, name)
            found.append((name, parent, mod, attr))
    return found


# ---------------------------------------------------------------------------
# Patch decoder in-place
# ---------------------------------------------------------------------------

def patch_decoder_with_mole(
    model: nn.Module,
    gate_store: GateStore,
    expert_cfgs: list[dict],
    target_modules: list[str] = ("q_proj", "v_proj"),
    dropout: float = 0.0,
) -> int:
    """
    Replace each target decoder Linear with a MoELoRALinear.

    expert_cfgs: ordered list [spa_cfg, cs_cfg, grn_cfg], each:
        adapter_path : str   path to lora_adapters.pt
        r            : int
        alpha        : float

    Returns the number of layers replaced.
    """
    targets = _find_target_linears(model, list(target_modules))
    log.info("Found %d target decoder linears (%s)", len(targets), ", ".join(target_modules))

    replaced = 0
    skipped: list[str] = []
    for full_name, parent, linear, attr in targets:
        mole = build_moe_lora_linear(
            linear=linear,
            gate_store=gate_store,
            module_name=full_name,
            expert_cfgs=expert_cfgs,
            dropout=dropout,
        )
        if mole is None:
            skipped.append(full_name)
            continue
        setattr(parent, attr, mole)
        replaced += 1

    if skipped:
        log.warning(
            "%d layers skipped (missing expert weights): %s ...",
            len(skipped),
            skipped[:3],
        )
    log.info("Replaced %d / %d decoder linears with MoELoRALinear", replaced, len(targets))
    return replaced


# ---------------------------------------------------------------------------
# Register encoder output hook → writes gates to GateStore
# ---------------------------------------------------------------------------

def _register_encoder_hook(
    model: nn.Module,
    router: SoftRegimeRouter,
    gate_store: GateStore,
) -> torch.utils.hooks.RemovableHook:
    """
    Attach a forward hook to model.encoder_proj so that every time the encoder
    projection fires, the router computes gates and stores them before the decoder.

    encoder outputs 2048-dim; encoder_proj projects to 4096-dim (decoder dim).
    We hook encoder_proj because:
      1. The router was trained on 4096-dim embeddings (captured at encoder_proj)
      2. 4096-dim captures richer representations than the raw 2048-dim encoder output
    """

    def _hook(module: nn.Module, input: Any, output: Any) -> None:
        # encoder_proj output: [B, T, 4096] or [T, 4096] tensor (fairseq2 Linear)
        enc_out = output[0] if isinstance(output, tuple) else output
        # Ensure float32 for router stability (model may be bfloat16)
        enc_out = enc_out.float()

        with torch.set_grad_enabled(router.training):
            gates = router(enc_out, padding_mask=None)
        gate_store.set(gates)

    hook_target = model.encoder_proj
    log.info("Hooking encoder_proj (2048→4096) for router input.")
    return hook_target.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_non_router(model: nn.Module) -> None:
    """
    Stage A2: freeze everything; only the router MLP is trainable.
    Call after patch_decoder_with_mole to ensure MoELoRALinear base + experts
    are also frozen.
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze MoELoRALinear expert weights — they stay frozen in A2
    # (router MLP is a separate module; caller must add it explicitly)


def freeze_for_a3(model: nn.Module) -> None:
    """
    Stage A3: freeze base model backbone; leave router + LoRA experts trainable.
    Expects model to already have MoELoRALinear layers.
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze LoRA expert A/B matrices
    for name, mod in model.named_modules():
        if isinstance(mod, MoELoRALinear):
            mod.unfreeze_experts()


# ---------------------------------------------------------------------------
# Main entry: build_jmole_model
# ---------------------------------------------------------------------------

def build_jmole_model(
    config: dict,
    stage: str = "a2",
) -> tuple[nn.Module, SoftRegimeRouter, GateStore]:
    """
    Load OmniASR, patch decoder with MoELoRALinear, attach router hook.

    Parameters
    ----------
    config : dict loaded from jmole_grn.yaml or jmole_spa.yaml
    stage  : "a2" (router only) | "a3" (router + LoRA)

    Returns
    -------
    model       : patched Wav2Vec2LlamaModel
    router      : SoftRegimeRouter (trainable MLP, separate nn.Module)
    gate_store  : GateStore shared between router and MoELoRALinear layers
    """
    from fairseq2.models.hub import load_model

    model_card   = config["model"]["card"]
    device_str   = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device       = torch.device(device_str)
    dtype_str    = config.get("dtype", "float32")
    _dtype_map   = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype        = _dtype_map.get(dtype_str, torch.float32)

    log.info("Loading base OmniASR model: %s on %s (%s)", model_card, device, dtype)
    model = load_model(model_card, device=torch.device("cpu"), dtype=dtype)

    # ── Build expert configs from yaml ────────────────────────────────────────
    expert_cfgs = []
    for key in ("spa", "cs", "grn"):
        ecfg = config["experts"][key]
        lora_cfg_path = Path(ecfg["lora_config"])
        lora_cfg = json.loads(lora_cfg_path.read_text())
        expert_cfgs.append({
            "adapter_path": ecfg["adapter_path"],
            "r":     lora_cfg["r"],
            "alpha": lora_cfg["lora_alpha"],
        })

    # ── Gate store ────────────────────────────────────────────────────────────
    gate_store = GateStore()

    # ── Patch decoder ─────────────────────────────────────────────────────────
    n_replaced = patch_decoder_with_mole(
        model=model,
        gate_store=gate_store,
        expert_cfgs=expert_cfgs,
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        dropout=config.get("lora_dropout", 0.0),
    )
    if n_replaced == 0:
        raise RuntimeError(
            "No decoder layers were replaced with MoELoRALinear. "
            "Check target_modules and adapter paths."
        )

    # ── Build router ──────────────────────────────────────────────────────────
    router_cfg = config.get("router", {})
    router = SoftRegimeRouter(
        d_enc   = router_cfg.get("d_enc", 4096),
        hidden  = router_cfg.get("hidden", 512),
        pooling = router_cfg.get("pooling", "mean_std"),
        dropout = router_cfg.get("dropout", 0.1),
        gate_store = gate_store,
    )

    # ── Register encoder output hook ──────────────────────────────────────────
    _register_encoder_hook(model, router, gate_store)
    log.info("Encoder hook registered: router will write gates before each decoder forward.")

    # ── Freeze according to training stage ───────────────────────────────────
    if stage == "a2":
        freeze_non_router(model)
        # Only router MLP parameters are trainable
        for p in model.parameters():
            p.requires_grad = False
        log.info("Stage A2: all model params frozen. Router MLP is trainable.")
    elif stage == "a3":
        freeze_for_a3(model)
        log.info("Stage A3: base model frozen; LoRA experts + router trainable.")

    # Freeze the encoder regardless of stage (always frozen)
    for name, p in model.named_parameters():
        if name.startswith("encoder") or name.startswith("encoder_frontend"):
            p.requires_grad = False

    # ── Move to device; cast only the LoRA A/B matrices to model dtype ───────
    # We do NOT call model.to(dtype=dtype) on the whole model because the
    # encoder's RoPE (view_as_complex) doesn't support bfloat16. load_model
    # already set the base model dtype correctly. We only need to cast the newly
    # added MoELoRALinear expert A/B matrices from their initial float32.
    model.to(device=device)
    for mod in model.modules():
        if isinstance(mod, MoELoRALinear):
            for key in EXPERT_KEYS:
                mod.A[key].to(device=device, dtype=dtype)
                mod.B[key].to(device=device, dtype=dtype)
    router.to(device=device)   # router stays float32 for numerical stability

    n_trainable_model  = sum(p.numel() for p in model.parameters()  if p.requires_grad)
    n_trainable_router = sum(p.numel() for p in router.parameters() if p.requires_grad)
    log.info(
        "Trainable params — model: %d, router: %d",
        n_trainable_model, n_trainable_router,
    )

    return model, router, gate_store


# ---------------------------------------------------------------------------
# One-hot gate injection (used by smoke_test_onehot_mole.py)
# ---------------------------------------------------------------------------

def set_onehot_gate(gate_store: GateStore, expert_idx: int, device: torch.device) -> None:
    """Inject a one-hot gate and lock it so the encoder hook cannot overwrite it."""
    g = torch.zeros(3, device=device)
    g[expert_idx] = 1.0
    gate_store.set_locked(g)
