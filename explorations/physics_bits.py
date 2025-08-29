from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from explorations.utils import load_cfg, load_model, make_example, out_dir
from src.model.rope2d import apply_rope_2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device=device)
    ex = make_example(cfg, device=device)
    H, W = cfg.board.H, cfg.board.W
    HxW = H * W
    # Segment ranges
    i = 0; i += 1; rule = (i, i + 18); i = rule[1]; i += 1; t = (i, i + HxW); i = t[1]; i += 1; t1 = (i, i + HxW)
    # Prepare RoPE hooks like training
    rotary_dim = cfg.model.d_head
    pos2d = ex.pos2d

    def _hq(q, hook):
        return apply_rope_2d(q, q, pos2d.to(q.device), rotary_dim=rotary_dim)[0]

    def _hk(k, hook):
        return apply_rope_2d(k, k, pos2d.to(k.device), rotary_dim=rotary_dim)[0]

    # Prepare hooks to apply RoPE consistently
    hooks = []
    for L in range(model.cfg.n_layers):
        hooks.append((f"blocks.{L}.attn.hook_q", _hq))
        hooks.append((f"blocks.{L}.attn.hook_k", _hk))
    out = out_dir()
    # For each layer/head, compute total attention from target queries to rule tokens
    attn_rule = np.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=np.float32)
    for L in range(model.cfg.n_layers):
        scores_name = f"blocks.{L}.attn.hook_attn_scores"
        box = {}
        def _cap_scores(t, hook):
            box["scores"] = t.detach().cpu()
        model.run_with_hooks(ex.tokens, return_type=None, fwd_hooks=hooks + [(scores_name, _cap_scores)])
        scores = box["scores"]  # [B,H,Q,K]
        s = scores[0]  # [H, Q, K]
        # Restrict queries to t1 segment and keys to rule segment
        s_t1 = s[:, t1[0]:t1[1], :]
        s_rule = s_t1[:, :, rule[0]:rule[1]]
        # Turn scores into weights (softmax over K) for magnitude invariance
        w = torch.softmax(torch.tensor(s_rule), dim=-1).numpy()
        attn_rule[L] = w.mean(axis=(1, 2))  # avg over Q and rule keys

    # Plot a heatmap of rule-attention by (layer, head)
    plt.figure(figsize=(6, 3))
    plt.imshow(attn_rule, aspect='auto', cmap='magma')
    plt.colorbar(); plt.xlabel('Head'); plt.ylabel('Layer'); plt.title('Avg attention to rule tokens')
    path = os.path.join(out, 'attn_to_rule_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print('saved:', path)


if __name__ == '__main__':
    main()
