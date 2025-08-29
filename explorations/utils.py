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
from src.data.stream import get_default_vocab


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
    with initialize(config_path=cfg_dir, version_base=None):
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


def build_trainlike_hooks(cfg: DictConfig, model, pos2d: torch.Tensor, tokens: torch.Tensor):
    # Recreate training-time hooks: 2D RoPE on Q/K, attention masking for t+1 queries,
    # and segment embedding at input if present.
    H, W = cfg.board.H, cfg.board.W
    rotary_dim = cfg.model.d_head

    def rope_single(x, pos2d_local):
        from src.model.rope2d import apply_rope_2d as rope
        x2, _ = rope(x, x, pos2d_local, rotary_dim=rotary_dim)
        return x2

    def _hq(q, hook):
        return rope_single(q, pos2d.to(q.device))

    def _hk(k, hook):
        return rope_single(k, pos2d.to(k.device))

    def _hs(scores, hook):
        # Restrict queries in target segments to attend only up to end of t segment (no leakage)
        B2, Hh, Q, K = scores.shape
        HxW = H * W
        start_t1 = 1 + 18 + 1 + HxW + 1
        allowed_k = torch.zeros((K,), dtype=torch.bool, device=scores.device)
        allowed_k[: start_t1 - 1] = True  # up to end of t
        q_in_target = torch.zeros((Q,), dtype=torch.bool, device=scores.device)
        q_in_target[start_t1:] = True
        bad = q_in_target[:, None] & (~allowed_k)[None, :]
        return scores.masked_fill(bad.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

    def _he(x, hook):
        # Add segment/type embeddings (0=physics/rule/seps, 1=state t, 2=targets)
        seg_table = getattr(model, 'segment_embed', None)
        if seg_table is None:
            return x
        B, T = x.size(0), x.size(1)
        HxW = H * W
        seg = torch.zeros((B, T), dtype=torch.long, device=x.device)
        i = 0
        i += 1; i += 18; i += 1
        seg[:, i : i + HxW] = 1
        i += HxW; i += 1
        seg[:, i : i + HxW] = 2
        i += HxW
        ms = int(getattr(getattr(cfg, 'train', {}), 'multi_steps', 0))
        if ms >= 2:
            i += 1; seg[:, i : i + HxW] = 2; i += HxW
            if ms >= 3:
                i += 1; seg[:, i : i + HxW] = 2; i += HxW
        return x + seg_table(seg)

    hooks = []
    for L in range(model.cfg.n_layers):
        hooks.append((f"blocks.{L}.attn.hook_q", _hq))
        hooks.append((f"blocks.{L}.attn.hook_k", _hk))
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", _hs))
    hooks.append(("hook_embed", _he))
    return hooks


def mask_targets_like_train(tokens: torch.Tensor, vocab: dict, cfg: DictConfig) -> torch.Tensor:
    # Replace t+1 (and later) positions with <MASK>, to match train-time inputs
    B, T = tokens.shape
    H, W = cfg.board.H, cfg.board.W
    HxW = H * W
    tks = tokens.clone()
    mask_id = int(vocab.get("<MASK>", vocab.get("0", 0)))
    i = 0
    i += 1; i += 18; i += 1
    i += HxW
    i += 1
    # t+1 segment
    tks[:, i : i + HxW] = mask_id
    i += HxW
    ms = int(getattr(getattr(cfg, 'train', {}), 'multi_steps', 0))
    if ms >= 2:
        i += 1; tks[:, i : i + HxW] = mask_id; i += HxW
        if ms >= 3:
            i += 1; tks[:, i : i + HxW] = mask_id; i += HxW
    return tks


def infer_logits_trainlike(model, cfg: DictConfig, tokens: torch.Tensor, pos2d: torch.Tensor, use_mask: bool = True):
    # Build hooks and optionally mask target inputs before running the model
    vocab = get_default_vocab()
    tokens_in = mask_targets_like_train(tokens, vocab, cfg) if use_mask else tokens
    hooks = build_trainlike_hooks(cfg, model, pos2d, tokens_in)
    return model.run_with_hooks(tokens_in, return_type="logits", fwd_hooks=hooks)


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
