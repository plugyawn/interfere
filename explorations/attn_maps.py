from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from explorations.utils import load_cfg, load_model, make_example, out_dir
from src.model.rope2d import apply_rope_2d


def get_segments(H: int, W: int):
    HxW = H * W
    i = 0
    i += 1  # BOS
    rule = (i, i + 18)
    i = rule[1]
    i += 1  # SEP
    t = (i, i + HxW)
    i = t[1]
    i += 1  # SEP2
    t1 = (i, i + HxW)
    return rule, t, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--center", action="store_true", help="Use center target cell query")
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device=device)
    ex = make_example(cfg, device=device)

    H, W = cfg.board.H, cfg.board.W
    rule_rng, t_rng, t1_rng = get_segments(H, W)
    # Choose a target query index (center of t1)
    if args.center:
        q_offset = (H // 2) * W + (W // 2)
    else:
        q_offset = 0
    q_index = t1_rng[0] + q_offset

    # Prepare RoPE hooks like training
    rotary_dim = cfg.model.d_head
    pos2d = ex.pos2d

    def rope_single(x, pos2d_local):
        x2, _ = apply_rope_2d(x, x, pos2d_local, rotary_dim=rotary_dim)
        return x2

    def _hq(q, hook):
        return rope_single(q, pos2d.to(q.device))

    def _hk(k, hook):
        return rope_single(k, pos2d.to(k.device))

    scores_name = f"blocks.{args.layer}.attn.hook_attn_scores"
    hooks = [
        (f"blocks.{args.layer}.attn.hook_q", _hq),
        (f"blocks.{args.layer}.attn.hook_k", _hk),
    ]
    # Capture attention scores via run_with_hooks
    box = {}
    def _capture_scores(t, hook):
        box["scores"] = t.detach().cpu()
    model.run_with_hooks(ex.tokens, return_type=None, fwd_hooks=hooks + [(scores_name, _capture_scores)])
    scores = box["scores"]  # [B,H,Q,K]

    head = args.head
    attn = scores[0, head]  # [Q, K]
    # Extract the query row for our target cell
    attn_q = attn[q_index]  # [K]

    # Map keys in t segment onto a heatmap
    t_start, t_end = t_rng
    t_scores = attn_q[t_start:t_end].view(H, W)

    out = out_dir()
    plt.figure(figsize=(4, 4))
    plt.imshow(t_scores, cmap="viridis")
    plt.colorbar(); plt.title(f"Layer {args.layer} Head {args.head} attn to t (query@t+1)")
    path = os.path.join(out, f"attn_L{args.layer}_H{args.head}_center{int(args.center)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("saved:", path)


if __name__ == "__main__":
    main()
