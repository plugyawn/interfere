from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import hydra
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from glob import glob

from src.data.stream import assemble_sequence, get_default_vocab, ToroidalNeighborhood
from src.data.rules import sb_to_bits
from src.model.hooked_life import build_model


@dataclass
class Example:
    tokens: torch.Tensor
    pos2d: torch.Tensor
    mask: torch.Tensor
    rb: torch.Tensor
    t: torch.Tensor
    t1: torch.Tensor
    vocab: dict


def load_cfg(config_name: str = "exp/life32") -> DictConfig:
    # Resolve cfg dir relative to repo root
    here = os.path.dirname(__file__)
    cfg_dir = os.path.join("..", "cfg")  # hydra.initialize requires relative path
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=config_name)
    return cfg


def _resolve_ckpt(run_id: Optional[str] = None, ckpt_path: Optional[str] = None) -> Optional[str]:
    # Explicit path wins
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    # Env or arg run_id
    rid = run_id or os.environ.get("RUN_ID")
    if rid:
        p = os.path.join("checkpoints", rid, "latest.pt")
        if os.path.exists(p):
            return p
    # Try newest run subdir
    cands = sorted(glob(os.path.join("checkpoints", "*", "latest.pt")), key=lambda p: os.path.getmtime(p), reverse=True)
    if cands:
        return cands[0]
    # Fallback root latest
    p = os.path.join("checkpoints", "latest.pt")
    return p if os.path.exists(p) else None


def load_model(cfg: DictConfig, device: torch.device | None = None, run_id: Optional[str] = None, ckpt_path: Optional[str] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(cfg, device=device)
    ckpt = _resolve_ckpt(run_id=run_id, ckpt_path=ckpt_path)
    if ckpt is None:
        print("Warn: no checkpoint found; using randomly initialized model")
    else:
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(state.get("model", state), strict=False)
        except Exception as e:
            print("Warn: failed to load full state:", e)
    print("Moving model to device: ", device)
    model = model.to(device)
    model.eval()
    return model


def make_example(cfg: DictConfig, device: Optional[torch.device] = None, batch_size: int = 1) -> Example:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = get_default_vocab()
    H, W = cfg.board.H, cfg.board.W
    # Rule: Conway S23/B3
    rb = sb_to_bits({2, 3}, {3}).to(device).unsqueeze(0).expand(batch_size, -1)
    # Random batch; optionally seed center with a blinker in the first example
    t = (torch.rand(batch_size, 1, H, W, device=device) < 0.5).to(torch.int64)
    if batch_size > 0:
        t[0:1, :, H // 2, W // 2 - 1 : W // 2 + 2] = 1
    neigh = ToroidalNeighborhood(device=device)
    counts = neigh.neighbors(t.to(torch.float32))
    from src.data.stream import _apply_rule

    t1 = _apply_rule(t, counts, rb)
    tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab)
    return Example(tokens=tokens, pos2d=pos2d, mask=mask, rb=rb, t=t, t1=t1, vocab=vocab)


def out_dir(prefix: str = "assets/explorations") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(prefix, ts)
    os.makedirs(path, exist_ok=True)
    print("out_dir:", os.path.abspath(path))
    return path
