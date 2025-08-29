from __future__ import annotations

import argparse
import os
from typing import List

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
from src.data.stream import ToroidalNeighborhood


def pca_2d(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD on covariance proxy
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2].T  # [D,2]
    Y = Xc @ comps  # [N,2]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
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
        # Next alive
        next_alive.append(ex.t1.detach().cpu().numpy()[:, 0].reshape(-1))

    F = np.concatenate(feats, axis=0)
    C = np.concatenate(nb_counts, axis=0)
    Y = np.concatenate(next_alive, axis=0)

    Z = pca_2d(F)
    out = out_dir()

    # Scatter colored by neighbor count
    plt.figure(figsize=(5,4))
    sc = plt.scatter(Z[:,0], Z[:,1], c=C, cmap='viridis', s=4, alpha=0.5)
    plt.colorbar(sc, label='neighbor count (t)')
    plt.title(f'Layer {L} resid_pre PCA colored by count')
    path1 = os.path.join(out, f'embed_L{L}_count_scatter.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()

    # Scatter colored by next alive
    plt.figure(figsize=(5,4))
    sc = plt.scatter(Z[:,0], Z[:,1], c=Y, cmap='coolwarm', s=4, alpha=0.5)
    plt.colorbar(sc, label='next alive (t+1)')
    plt.title(f'Layer {L} resid_pre PCA colored by next-alive')
    path2 = os.path.join(out, f'embed_L{L}_next_scatter.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()

    print('saved:', path1)
    print('saved:', path2)


if __name__ == '__main__':
    main()

