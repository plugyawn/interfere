from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import imageio.v2 as imageio
from PIL import Image
import torch

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
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
    ap.add_argument("--annotate", action="store_true", help="Overlay 3x3 box and top-k locs")
    ap.add_argument("--topk", type=int, default=5, help="Top-k attention keys to mark")
    ap.add_argument("--sweep", choices=["none","heads","layers"], default="none", help="Animate across heads or layers")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
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

    def save_single(layer:int, head:int, arr:torch.Tensor, annotate:bool) -> str:
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(arr, cmap="viridis")
        fig.colorbar(im)
        ax.set_title(f"Layer {layer} Head {head} → t (q@t+1)")
        if annotate:
            cy, cx = H//2, W//2
            # 3x3 box around center
            rect = patches.Rectangle((cx-1.5, cy-1.5), 3, 3, linewidth=1.5, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            # mark top-k keys
            flat = arr.detach().cpu().view(-1)
            topk = min(args.topk, flat.numel())
            vals, idxs = torch.topk(flat, k=topk)
            for i, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist())):
                y, x = divmod(idx, W)
                ax.plot(x, y, marker='o', markersize=3, color='red', alpha=0.9)
        path = os.path.join(out, f"attn_L{layer}_H{head}_center{int(args.center)}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # Save single annotated map
    single_path = save_single(args.layer, head, t_scores, args.annotate)
    print("saved:", single_path)

    # Optional sweep animation
    if args.sweep != "none":
        frames: List[str] = []
        if args.sweep == "heads":
            for hh in range(scores.shape[1]):
                arr = scores[0, hh, q_index, t_start:t_end].view(H, W)
                frames.append(save_single(args.layer, hh, arr, args.annotate))
            anim_path = os.path.join(out, f"attn_sweep_heads_L{args.layer}_center{int(args.center)}.gif")
        else:  # layers
            nL = model.cfg.n_layers
            for L in range(nL):
                # recalc scores per layer
                scores_name2 = f"blocks.{L}.attn.hook_attn_scores"
                box2 = {}
                def _cap2(t, hook):
                    box2["scores"] = t.detach().cpu()
                hooks2 = [
                    (f"blocks.{L}.attn.hook_q", _hq),
                    (f"blocks.{L}.attn.hook_k", _hk),
                ]
                model.run_with_hooks(ex.tokens, return_type=None, fwd_hooks=hooks2 + [(scores_name2, _cap2)])
                sc = box2["scores"][0, head]
                arr = sc[q_index, t_start:t_end].view(H, W)
                frames.append(save_single(L, head, arr, args.annotate))
            anim_path = os.path.join(out, f"attn_sweep_layers_H{head}_center{int(args.center)}.gif")

        # Write GIF with uniform frame size
        raw = [Image.open(p).convert('RGB') for p in frames]
        w = min(im.size[0] for im in raw); h = min(im.size[1] for im in raw)
        imgs = [im.resize((w, h)) for im in raw]
        imageio.mimsave(anim_path, imgs, duration=max(0.01, 1.0/args.fps))
        print("saved animation:", anim_path)


if __name__ == "__main__":
    main()
