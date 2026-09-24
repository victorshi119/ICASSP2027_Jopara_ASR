#!/usr/bin/env python3
"""
Print all Linear module names under the LLaMA decoder so you can set target_modules
for LoRA (e.g. q_proj, v_proj or q/k/v/o). Run with same env as training (OMNILINGUAL_ASR_ROOT).
"""
import argparse
import sys
from pathlib import Path


def print_decoder_linear_modules(model_card: str, checkpoint_path: Path = None):
    try:
        from fairseq2.models.llama import LLaMAConfig
        from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
    except ImportError:
        print("Import error: set PYTHONPATH to include omnilingual-asr/src and fairseq2.")
        return
    import torch
    try:
        from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub
        hub = get_wav2vec2_llama_model_hub()
        config = hub.get_model_config(model_card)
        if checkpoint_path and checkpoint_path.exists():
            model = hub.load_custom_model(checkpoint_path, config, device="cpu", dtype=torch.float32)
        else:
            model = hub.load_model(model_card, device="cpu", dtype=torch.float32)
    except Exception as e:
        print("Load failed:", e)
        return
    for name, mod in model.named_modules():
        if "Linear" in type(mod).__name__ or "linear" in name.lower():
            if "llama_decoder" in name:
                print(name)
    print("(Use these names for target_modules; match q_proj, v_proj, k_proj, o_proj if present.)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-card", type=str, default="omniASR_LLM_7B_v2")
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()
    print_decoder_linear_modules(args.model_card, args.checkpoint)


if __name__ == "__main__":
    main()
