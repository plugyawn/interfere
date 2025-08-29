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
    ap.add_argument("--mode", choices=["scores","weights","contrib"], default="weights", help="What to visualize from the head")
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
    # Capture attention scores and resid_pre via run_with_hooks/run_with_cache
    box = {}
    def _capture_scores(t, hook):
        box["scores"] = t.detach().cpu()
    name_resid = f"blocks.{args.layer}.hook_resid_pre"
    def _cap_resid(x, hook):
        box["resid_pre"] = x.detach().cpu()
    model.run_with_hooks(ex.tokens, return_type=None, fwd_hooks=hooks + [(scores_name, _capture_scores), (name_resid, _cap_resid)])
    scores = box["scores"]  # [B,H,Q,K]
    resid_pre = box.get("resid_pre")  # [B,T,D] (may be None if not captured)

    head = args.head
    attn = scores[0, head]  # [Q, K]
    # Extract the query row for our target cell
    attn_q = attn[q_index]  # [K]

    # Map keys in t segment onto a heatmap
    t_start, t_end = t_rng
    # Choose visualization mode
    if args.mode == "scores":
        vec = attn_q
    elif args.mode == "weights":
        vec = torch.softmax(attn_q, dim=-1)
    else:  # contrib
        # Head OV contribution to alive logit per key: (resid_pre_i @ W_V @ W_O @ W_U[:, alive])
        assert resid_pre is not None, "resid_pre not captured"
        W_V = model.W_V[args.layer, head]  # [d_model, d_head] in TL
        W_O = model.W_O[args.layer, head]  # [d_head, d_model]
        W_U = model.W_U  # [d_model, d_vocab]
        alive_id = 1 if "ALIVE" in ex.vocab else 1
        ov = (W_V @ W_O).detach()  # [d_model, d_model]
        dir_alive = W_U[:, alive_id].detach()  # [d_model]
        R = resid_pre[0].detach()  # [T, d_model]
        contrib = (R @ ov) @ dir_alive  # [T]
        vec = attn_q * contrib  # elementwise per key position
    t_scores = vec[t_start:t_end].view(H, W)

    out = out_dir()

    def save_single(layer:int, head:int, arr:torch.Tensor, annotate:bool) -> str:
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(arr.detach().cpu(), cmap="viridis")
        fig.colorbar(im)
        ax.set_title(f"Layer {layer} Head {head} → t (q@t+1) [{args.mode}]")
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

    # Helper to build map per (L,H) per mode
    def build_map_for(layer_i:int, head_i:int, scores_tensor:torch.Tensor, resid_tensor:torch.Tensor|None):
        attn = scores_tensor[0, head_i]  # [Q,K]
        vec_local = attn[q_index]
        if args.mode == "scores":
            v = vec_local
        elif args.mode == "weights":
            v = torch.softmax(vec_local, dim=-1)
        else:
            assert resid_tensor is not None, "resid_pre not captured for contrib"
            W_V = model.W_V[layer_i, head_i]
            W_O = model.W_O[layer_i, head_i]
            W_U = model.W_U
            alive_id = 1
            ov = W_V @ W_O
            dir_alive = W_U[:, alive_id]
            R = resid_tensor[0]
            contrib = (R @ ov) @ dir_alive
            v = vec_local * contrib
        return v[t_start:t_end].view(H, W)

    # Optional sweep animation
    if args.sweep != "none":
        frames: List[str] = []
        if args.sweep == "heads":
            for hh in range(scores.shape[1]):
                arr = build_map_for(args.layer, hh, scores, resid_pre)
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
                name_resid2 = f"blocks.{L}.hook_resid_pre"
                box2_res = {}
                def _cap_res2(x, hook):
                    box2_res["resid_pre"] = x.detach().cpu()
                hooks2 = [
                    (f"blocks.{L}.attn.hook_q", _hq),
                    (f"blocks.{L}.attn.hook_k", _hk),
                ]
                cap_hooks = hooks2 + [(scores_name2, _cap2)]
                if args.mode == "contrib":
                    cap_hooks += [(name_resid2, _cap_res2)]
                model.run_with_hooks(ex.tokens, return_type=None, fwd_hooks=cap_hooks)
                sc = box2["scores"]
                resL = box2_res.get("resid_pre") if args.mode == "contrib" else None
                arr = build_map_for(L, head, sc, resL)
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
