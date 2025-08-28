from __future__ import annotations

import sys

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.train.loop import train_loop_ddp


def main():
    exp_name = None
    # Usage: python ddp_train.py [life32|life_smoke]
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("life32", "life_smoke", "life24", "life40"):
            exp_name = arg
    if exp_name is None:
        exp_name = "life32"

    with initialize(config_path="cfg"):
        cfg = compose(config_name=f"exp/{exp_name}")
    # Hydra compose here nests content under 'exp'; unwrap for trainer
    cfg_unwrapped = cfg.get("exp", cfg)
    print("Composed cfg:\n" + OmegaConf.to_yaml(cfg_unwrapped)[:400])
    # Use manual cfg for robustness in DDP launch
    import types
    mc = types.SimpleNamespace(
        n_layers=8, d_model=512, n_heads=8, d_head=64, d_mlp=2048, n_ctx=2100, attn_impl="sdpa", film=False
    )
    tc = types.SimpleNamespace(
        bf16=True,
        compile=types.SimpleNamespace(enabled=False, scope="block"),
        grad_ckpt=types.SimpleNamespace(mlp_only=True),
        batch_per_gpu=8,
        grad_accum=1,
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=200,
        steps=None,
        target_tokens=2.0e7,
        seed=123,
        fast=True,
        multi_steps=0,
    )
    board = types.SimpleNamespace(H=32, W=32)
    cfg2 = types.SimpleNamespace(model=mc, train=tc, board=board)
    stats = train_loop_ddp(cfg2)
    print({"loss": stats.loss, "acc": stats.acc, "tokens_per_step": stats.tokens_per_step, "tps": stats.tps, "secs": stats.secs})


if __name__ == "__main__":
    main()
