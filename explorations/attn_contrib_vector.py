from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import load_cfg, load_model, make_example, out_dir, build_trainlike_hooks, mask_targets_like_train
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
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device=device, run_id=args.run_id, ckpt_path=args.ckpt)
    ex = make_example(cfg, device=device)

    H, W = cfg.board.H, cfg.board.W
    HxW = H * W
    rule_rng, t_rng, t1_rng = get_segments(H, W)
    # center query in t1
    q_index = t1_rng[0] + (H // 2) * W + (W // 2)

    # Capture scores + resid with train-like hooks
    L = int(args.layer)
    head = int(args.head)
    scores_name = f"blocks.{L}.attn.hook_attn_scores"
    name_resid = f"blocks.{L}.hook_resid_pre"
    box = {}
    def _cap_scores(t, hook):
        box["scores"] = t.detach().cpu()
    def _cap_resid(x, hook):
        box["resid_pre"] = x.detach().cpu()
    tokens_in = mask_targets_like_train(ex.tokens, ex.vocab, cfg)
    hooks = build_trainlike_hooks(cfg, model, ex.pos2d, tokens_in)
    model.run_with_hooks(tokens_in, return_type=None, fwd_hooks=hooks + [(scores_name, _cap_scores), (name_resid, _cap_resid)])
    scores = box["scores"][0, head]  # [Q,K]
    resid_pre = box["resid_pre"][0]

    # Contribution map for this head
    attn_q = scores[q_index]  # [K]
    W_V = model.W_V[L, head]
    W_O = model.W_O[L, head]
    W_U = model.W_U
    alive_id = 1
    ov = (W_V @ W_O).detach()
    dir_alive = W_U[:, alive_id].detach()
    contrib = (resid_pre @ ov) @ dir_alive  # [T]
    vec = attn_q * contrib  # [K]
    t_start, t_end = t_rng
    arr = vec[t_start:t_end].view(H, W).detach().cpu().numpy()

    # Compute gradient-based vector field
    gy, gx = np.gradient(arr)

    out = out_dir()
    plt.figure(figsize=(6,5))
    plt.imshow(arr, cmap='viridis')
    plt.colorbar(label='contrib')
    step = max(1, min(H,W)//16)
    Y, X = np.mgrid[0:H:1, 0:W:1]
    plt.quiver(X[::step,::step], Y[::step,::step], gx[::step,::step], gy[::step,::step], color='white', scale=50)
    plt.title(f'Layer {L} Head {head}: contrib heatmap + gradient vectors')
    path = os.path.join(out, f'contrib_vector_L{L}_H{head}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print('saved:', path)


if __name__ == '__main__':
    main()

