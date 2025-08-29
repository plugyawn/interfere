from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import hydra
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

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


def load_model(cfg: DictConfig, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(cfg)
    ckpt = os.path.join("checkpoints", "latest.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(state.get("model", state), strict=False)
        except Exception as e:
            print("Warn: failed to load full state:", e)
    model = model.to(device)
    model.eval()
    return model


def make_example(cfg: DictConfig, device: torch.device | None = None) -> Example:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = get_default_vocab()
    H, W = cfg.board.H, cfg.board.W
    # Rule: Conway S23/B3
    rb = sb_to_bits({2, 3}, {3}).to(device).unsqueeze(0)
    # Make a random batch with a blinker centered
    t = (torch.rand(1, 1, H, W, device=device) < 0.5).to(torch.int64)
    t[:, :, H // 2, W // 2 - 1 : W // 2 + 2] = 1
    neigh = ToroidalNeighborhood(device=device)
    counts = neigh.neighbors(t.to(torch.float32))
    from src.data.stream import _apply_rule

    t1 = _apply_rule(t, counts, rb.expand(t.size(0), -1))
    tokens, mask, pos2d = assemble_sequence(rb.expand(t.size(0), -1), t, t1, vocab=vocab)
    return Example(tokens=tokens, pos2d=pos2d, mask=mask, rb=rb, t=t, t1=t1, vocab=vocab)


def out_dir(prefix: str = "assets/explorations") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(prefix, ts)
    os.makedirs(path, exist_ok=True)
    print("out_dir:", os.path.abspath(path))
    return path
