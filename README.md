# Life-GPT: Rule-Conditioned Cellular Automata with TransformerLens

This repo trains a TransformerLens-based model (HookedTransformer) to predict the next state of Conway-style cellular automata, conditioned on the S/B rule bits. It also provides lightweight mech‑interp utilities (probes, activation patching, AMA API).

## Quickstart

- Python >= 3.10 recommended.
- Single node with NVIDIA GPUs (Hopper/H200 ideal).

### 1) Install

Option A: Default install (CPU or preconfigured CUDA)

```
make setup
```

Option B: Install PyTorch with a specific CUDA wheel via index URL, then install deps

```
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
make setup
```

### 2) GPU sanity check

```
python -c "import torch; print({'cuda':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'bf16':torch.cuda.is_bf16_supported()})"
```

### 3) Run tests (none yet)

```
pytest -q || true
```

## Make targets

- `make setup`: Install dependencies (uses `TORCH_INDEX_URL` if set for CUDA wheels).
- `make test`: Run pytest.
- `make train`: Placeholder; runs `train.py` (added in S1+).
- `make eval`: Placeholder; runs `eval.py` (added in S1+).
- `make interp`: Placeholder; runs `interp_run.py` (added in S1+).

## Project Roadmap (S0 → S13)

We implement iteratively: bootstrap env, data, model, training, DDP, evaluation, and mech‑interp AMA API. Each step is committed with JSON metrics in the message.

