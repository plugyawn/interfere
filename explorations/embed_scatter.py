from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import load_cfg, load_model, make_example, out_dir, build_trainlike_hooks, mask_targets_like_train
from src.data.stream import ToroidalNeighborhood


def pca(X: np.ndarray, dim: int = 2) -> np.ndarray:
    X = X.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:dim].T  # [D,dim]
    return Xc @ comps  # [N,dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, choices=[2,3], default=2)
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

    H, W = cfg.board.H, cfg.board.W
    HxW = H * W
    L = int(args.layer)

    feats: List[np.ndarray] = []
    nb_counts: List[np.ndarray] = []
    next_alive: List[np.ndarray] = []
    curr_alive: List[np.ndarray] = []

    total = int(args.samples)
    bs = max(1, int(args.batch_size))
    for start in range(0, total, bs):
        cur_bs = min(bs, total - start)
        ex = make_example(cfg, device=device, batch_size=cur_bs)
        # Build train-like hooks + capture resid_pre
        tokens_in = mask_targets_like_train(ex.tokens, ex.vocab, cfg)
        hooks = build_trainlike_hooks(cfg, model, ex.pos2d, tokens_in)
        name = f"blocks.{L}.hook_resid_pre"
        box = {}
        def _cap(x, hook):
            box["x"] = x.detach().to(torch.float32).cpu()
        hooks2 = hooks + [(name, _cap)]
        model.run_with_hooks(tokens_in, return_type=None, fwd_hooks=hooks2)
        X = box["x"][..., :].cpu()  # [B,T,D]
        # Slice t segment
        t_start = 1 + 18 + 1
        X_t = X[:, t_start : t_start + HxW, :]  # [B,HxW,D]
        feats.append(X_t.reshape(-1, X_t.size(-1)).numpy())
        # Neighbor counts on t
        neigh = ToroidalNeighborhood(device=device)
        counts = neigh.neighbors(ex.t.to(torch.float32))  # [B,1,H,W]
        nb_counts.append(counts.detach().cpu().numpy()[:, 0].reshape(-1))
        # Next alive and current alive
        next_alive.append(ex.t1.detach().cpu().numpy()[:, 0].reshape(-1))
        curr_alive.append(ex.t.detach().cpu().numpy()[:, 0].reshape(-1))

    F = np.concatenate(feats, axis=0)
    C = np.concatenate(nb_counts, axis=0)
    Y = np.concatenate(next_alive, axis=0)
    Yc = np.concatenate(curr_alive, axis=0)

    Z = pca(F, dim=args.pca_dim)
    out = out_dir()

    def scatter2d(Z2: np.ndarray, color_vals: np.ndarray, cmap: str, title: str, cbar_label: str,
                  annotate_centroids: List[tuple] | None, path: str):
        plt.figure(figsize=(5,4))
        sc = plt.scatter(Z2[:,0], Z2[:,1], c=color_vals, cmap=cmap, s=4, alpha=0.5)
        plt.colorbar(sc, label=cbar_label)
        plt.title(title)
        if annotate_centroids:
            for mask, text, box_fc, text_col in annotate_centroids:
                if np.any(mask):
                    cx, cy = Z2[mask,0].mean(), Z2[mask,1].mean()
                    plt.text(cx, cy, text, color=text_col, ha='center', va='center', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', fc=box_fc, alpha=0.4))
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    def scatter3d(Z3: np.ndarray, color_vals: np.ndarray, cmap: str, title: str, cbar_label: str,
                  annotate_centroids: List[tuple] | None, path: str):
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=color_vals, cmap=cmap, s=4, alpha=0.5)
        fig.colorbar(p, ax=ax, label=cbar_label)
        ax.set_title(title)
        if annotate_centroids:
            for mask, text, box_fc, text_col in annotate_centroids:
                if np.any(mask):
                    cx, cy, cz = Z3[mask,0].mean(), Z3[mask,1].mean(), Z3[mask,2].mean()
                    ax.text(cx, cy, cz, text, color=text_col)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Build centroid annotations
    ann_count = [(C == k, str(k), 'black', 'white') for k in range(9)]
    ann_next = [((Y == 0), 'D', 'gray', 'black'), ((Y == 1), 'A', 'gray', 'white')]
    ann_curr = [((Yc == 0), 'D', 'gray', 'black'), ((Yc == 1), 'A', 'gray', 'white')]

    if args.pca_dim == 2:
        path1 = os.path.join(out, f'embed_L{L}_count_scatter.png')
        scatter2d(Z, C, 'viridis', f'Layer {L} resid_pre PCA colored by count', 'neighbor count (t)', ann_count, path1)
        path2 = os.path.join(out, f'embed_L{L}_next_scatter.png')
        scatter2d(Z, Y, 'coolwarm', f'Layer {L} resid_pre PCA colored by next-alive', 'next alive (t+1)', ann_next, path2)
        path3 = os.path.join(out, f'embed_L{L}_curr_scatter.png')
        scatter2d(Z, Yc, 'coolwarm', f'Layer {L} resid_pre PCA colored by current-alive', 'current alive (t)', ann_curr, path3)
    else:
        path1 = os.path.join(out, f'embed_L{L}_count_scatter_3d.png')
        scatter3d(Z, C, 'viridis', f'Layer {L} resid_pre PCA(3D) colored by count', 'neighbor count (t)', ann_count, path1)
        path2 = os.path.join(out, f'embed_L{L}_next_scatter_3d.png')
        scatter3d(Z, Y, 'coolwarm', f'Layer {L} resid_pre PCA(3D) colored by next-alive', 'next alive (t+1)', ann_next, path2)
        path3 = os.path.join(out, f'embed_L{L}_curr_scatter_3d.png')
        scatter3d(Z, Yc, 'coolwarm', f'Layer {L} resid_pre PCA(3D) colored by current-alive', 'current alive (t)', ann_curr, path3)

    print('saved:', path1)
    print('saved:', path2)
    print('saved:', path3)


if __name__ == '__main__':
    main()
