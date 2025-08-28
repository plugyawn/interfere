# interfere

interfere is a research codebase for training and analyzing small transformers with a focus on clarity, reliability, and multi‑GPU ergonomics. It builds on TransformerLens while providing:

- Clean training loops with DDP, token‑budget scheduling, tqdm progress, and periodic rollout artifacts.
- Hooks‑based extensions such as 2‑D rotary position encoding (RoPE) applied to Q/K.
- Simple, reproducible Hydra configurations and smoke‑testable defaults.
- Analysis‑friendly choices (explicit positional setup, shifted next‑token loss, asserts to catch common pitfalls).

The repository ships a demonstration task based on Conway’s Game of Life (GoL): conditioning on rule bits and predicting next‑state boards, with optional multi‑step supervision and rollout visualization. The same infrastructure is intended to generalize to other structured sequence tasks.

## Vision: towards a small, general library

While this repo centers a concrete demo, our longer‑term plan is to extract a small, general library that improves developer experience around TransformerLens:

- Ergonomic multi‑GPU training primitives: token‑budget scheduling, rank‑0 checkpointing, clean progress/logging.
- Safer analysis defaults: explicit positional controls, predictable hooks, and consistency checks that prevent subtle leakage.
- Lightweight visualization utilities: periodic rollout videos/images and summary artifacts.

If you’re using TransformerLens and want focused training/analysis helpers without a heavy framework, interfere aims to offer a well‑factored foundation you can adopt piecemeal.

## Demonstration: GoL modeling

We include a compact GoL setup as a worked example:

- Data stream: toroidal neighborhoods via `conv2d`, on‑device sampling, and optional canonical pattern injection.
- Tokenization: prefix rule bits, board tokens, and loss masks that supervise target segments only.
- Model: HookedTransformer with 2‑D RoPE (applied via hooks to Q/K), FiLM conditioning (optional), and bf16 training.
- Training: single‑GPU and DDP loops, shifted next‑token loss (no target leakage), optional multi‑target (t+2, t+3).
- Curriculum: scheduled sampling (annealed) to reduce brittleness from teacher‑forced inputs.
- Eval & viz: accuracy, rollout divergence, and periodic autoregressive rollout MP4s saved under `assets/rollouts/`.

## Quick start

### Install

```
make setup
```

Set a CUDA wheel explicitly if needed:

```
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
make setup
```

### Sanity check

```
python -c "import torch; print({'cuda':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'bf16':torch.cuda.is_bf16_supported()})"
```

### Run

- Single‑GPU smoke (16×16):
  - `python train.py --config-name exp/life_smoke train.fast=false train.steps=10 hydra.output_subdir=null hydra.run.dir=.`
- 8× DDP (32×32):
  - `torchrun --standalone --nproc_per_node=8 train.py --config-name exp/life32`
- 8× DDP with t+3 supervision:
  - `torchrun --standalone --nproc_per_node=8 train.py --config-name exp/life32_t3`

Artifacts:

- Checkpoints: `checkpoints/latest.pt` (rank‑0).
- Rollouts (MP4): `assets/rollouts/` at step 1 and every 50 steps.
- Calibration: `runs/calibration.json` (tokens/step and throughput snapshot).

## Notes on analysis hygiene

- We set TransformerLens `positional_embedding_type='standard'` and apply 2‑D RoPE via hooks — no mixed schemes.
- Loss is computed with a one‑step shift to prevent a token predicting itself or reading its own target.
- Asserts guard sequence length (`n_ctx`) and mask alignment; DDP uses device‑aligned RoPE hooks.

## Status

This codebase is evolving and aims to be readable, testable, and practical for small‑scale research. Contributions around multi‑GPU ergonomics and analysis safety are welcome as we factor utilities into a general helper library.
