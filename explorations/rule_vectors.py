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
from explorations.utils import load_cfg, load_model, out_dir
from src.data.stream import get_default_vocab


def pca_2d(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2].T
    return Xc @ comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
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
    vocab = get_default_vocab()

    # Collect embeddings for rule tokens and their differences
    # Cast to float32 before numpy (numpy has no bfloat16)
    E = model.W_E.detach().to(torch.float32).cpu().numpy()  # [V,D]
    rule0_ids = []
    rule1_ids = []
    for r in range(18):
        rule0_ids.append(int(vocab[f"<R{r}_0>"]))
        rule1_ids.append(int(vocab[f"<R{r}_1>"]))
    R0 = E[rule0_ids]  # [18,D]
    R1 = E[rule1_ids]  # [18,D]
    DV = R1 - R0       # [18,D]  direction vectors per rule bit

    # Cosine similarity heatmap between rule vectors
    def cos(a,b):
        n = (a * b).sum(-1)
        da = np.linalg.norm(a, axis=-1)
        db = np.linalg.norm(b, axis=-1)
        return n / np.maximum(1e-9, np.outer(da, db))
    C = cos(DV, DV)

    out = out_dir()
    # Heatmap annotated by S (0..8) vs B (9..17)
    plt.figure(figsize=(5,4))
    plt.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='cosine')
    plt.title('Rule vector cosine similarity (E[R1]-E[R0])')
    plt.xlabel('r (0..8=S, 9..17=B)'); plt.ylabel('r (0..8=S, 9..17=B)')
    path_cos = os.path.join(out, 'rule_vector_cosine.png')
    plt.savefig(path_cos, dpi=150, bbox_inches='tight'); plt.close()

    # PCA scatter of difference vectors, colored by S vs B and annotated by r
    Z = pca_2d(DV)
    labels = np.array([('S' if r < 9 else 'B') for r in range(18)])
    colors = np.where(labels=='S', '#1f77b4', '#d62728')
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], c=colors, s=40, alpha=0.9)
    for r,(x,y) in enumerate(Z):
        plt.text(x, y, f"{labels[r]}{r%9}", ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    plt.title('Rule bit difference vectors (PCA)')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    path_pca = os.path.join(out, 'rule_vectors_pca.png')
    plt.savefig(path_pca, dpi=150, bbox_inches='tight'); plt.close()

    # Arrows from R0 to R1 in PCA space (token embeddings themselves)
    Z_tokens = pca_2d(np.concatenate([R0, R1], axis=0))
    Z0 = Z_tokens[:18]
    Z1 = Z_tokens[18:]
    plt.figure(figsize=(5,4))
    for r in range(18):
        x0,y0 = Z0[r]
        x1,y1 = Z1[r]
        col = '#1f77b4' if r<9 else '#d62728'
        plt.arrow(x0, y0, x1-x0, y1-y0, color=col, head_width=0.05, length_includes_head=True, alpha=0.8)
        plt.text(x1, y1, f"{('S' if r<9 else 'B')}{r%9}", fontsize=7, color=col)
    plt.title('Rule token shift: <Rr_0> → <Rr_1> (PCA of embeddings)')
    path_arrow = os.path.join(out, 'rule_token_arrows.png')
    plt.savefig(path_arrow, dpi=150, bbox_inches='tight'); plt.close()

    print('saved:', path_cos)
    print('saved:', path_pca)
    print('saved:', path_arrow)


if __name__ == '__main__':
    main()
