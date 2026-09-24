# Training

Everything here fine-tunes on top of a **frozen** OmniASR LLM 7B v2 (`omniASR_LLM_7B_v2`). It
needs `omnilingual-asr` with our patch applied (see `third_party/`).

## LoRA adapters (`lora_adapters/`)

`train_lora.py` is a thin wrapper around the fairseq2 training recipe. It injects LoRA into the
decoder's query and value projections, trains, and saves only the adapter weights. The config
files set the hyperparameters. `batches/train_paper_adapters.sh` lists the exact runs:

| Run | Used as | Config | Training data | Language mode |
|---|---|---|---|---|
| `B2_spa_d010` | Table 1 row 5; Spanish expert in rows 8–9 | `b2_spa_d010.yaml` (r 16, lr 1e-4) | 1,083 Spanish-only | `spa_Latn` |
| `B3_grn_cv_cegpa` | Table 1 row 6 | `b3_grn_base.yaml` (r 16, lr 2e-5) | 2,569 Common Voice + 19 Guaraní-only + 871 code-switching | `grn_Latn` |
| `B2_cs_r32_spa` | Table 1 row 7 | `b2_cs_r32.yaml` (r 32, lr 1e-4) | 871 code-switching | `spa_Latn` |
| `B2_cs_r32_grn` | Code-switching expert in row 8 | `b2_cs_r32.yaml` | 871 code-switching | `grn_Latn` |
| `B3_grn_cv_cegpa_seed{13,42,123}` | Guaraní expert in rows 8–9, one per seed | `b3_grn_base.yaml` | same as row 6 | `grn_Latn` |

All runs use dropout 0.10, LoRA alpha = 2 × rank, AdamW with a cosine schedule, bf16, and up to
4,000 steps with early stopping on development error. The training data column refers to the
split files described in `data/README.md`.

Other files:

- `make_full_7b_manifest.py` turns a split CSV into the fairseq2 manifest.
- `download_tokenizer.py` pre-fetches the tokenizer.
- `extract_lora_adapter.py` recovers `lora_adapters.pt` from a run's checkpoints.
- `batches/run_lora_v3.sbatch` and `batches/run_lora_b3.sbatch` are the SLURM launchers. They
  also write the dataset card that `omnilingual-asr` needs.

## Mixture-of-experts router (`moe_router/`)

The router reads pooled encoder output and gives each utterance soft weights over three experts:
Spanish, Guaraní, and code-switching.

1. `extract_encoder_embeddings.py` caches pooled encoder embeddings for the train and dev sets.
2. `build_soft_regime_targets.py` builds the router's soft targets. Spanish-only and Guaraní-only
   utterances get 0.90 on their own expert. Code-switched utterances get 0.70 on the
   code-switching expert, and the rest is split by word-level evidence.
3. `train_router_on_embeddings.py` trains the router with the adapters frozen (KL loss plus a
   small entropy term). `batches/run_router_seeds.sbatch` trains seeds 13, 42, and 123.
4. `infer_jmole_asr.py` decodes the test set with the router and the three experts, all in the
   Guaraní language mode. The launchers are `batches/run_row8_eval_seed.sbatch` and
   `batches/run_a2_nullcs_eval_seed.sbatch`, with one config per seed in `configs/`.

- **Row 8** uses `configs/row8_seed*.yaml`: experts `B2_spa_d010`, `B3_grn_cv_cegpa_seed*`, and
  `B2_cs_r32_grn`.
- **Row 9** uses `configs/a2_nullcs_router_seed*.yaml`: the same, except that the code-switching
  slot holds an all-zero adapter (`null_cs_z0`, same shape as the others). That slot is therefore
  the base model in the Guaraní language mode.

Both rows use the same router checkpoints. In each seed, the router *and* the Guaraní expert
change together.

`model/` holds the router and MoE-LoRA modules that `infer_jmole_asr.py` patches into the
OmniASR decoder. `configs/branch_registry.yaml` maps each expert name to its adapter file,
rank, and language mode.

## Paths

Python scripts read data from `$JOPARA_ROOT`, which defaults to our cluster path and follows the
original project layout. The `.sbatch` files and YAML configs still contain our cluster's absolute
paths (`/N/project/...`, `/N/scratch/...`, `/N/slate/...`). They are kept exactly as they ran, and
you will need to edit those paths.
