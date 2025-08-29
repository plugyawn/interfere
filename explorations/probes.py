from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import load_cfg, load_model, make_example, out_dir
from src.model.rope2d import apply_rope_2d


def collect_resid_pre(model, tokens, pos2d, layer: int):
    rotary_dim = model.cfg.d_head

    def _hq(q, hook):
        return apply_rope_2d(q, q, pos2d.to(q.device), rotary_dim=rotary_dim)[0]

    def _hk(k, hook):
        return apply_rope_2d(k, k, pos2d.to(k.device), rotary_dim=rotary_dim)[0]

    name = f"blocks.{layer}.hook_resid_pre"
    box = {}

    def _cap(x, hook):
        # Move to CPU float32 for numpy compatibility
        box["x"] = x.detach().to(torch.float32).cpu()

    # Apply RoPE on all layers to match training geometry
    fwd_hooks = []
    for L in range(model.cfg.n_layers):
        fwd_hooks.append((f"blocks.{L}.attn.hook_q", _hq))
        fwd_hooks.append((f"blocks.{L}.attn.hook_k", _hk))
    fwd_hooks.append((name, _cap))

    model.run_with_hooks(tokens, return_type=None, fwd_hooks=fwd_hooks)
    return box["x"]  # [B, T, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device=device)
    H, W = cfg.board.H, cfg.board.W
    HxW = H * W
    out = out_dir()

    # Build a small dataset
    X_by_layer = {}
    y_count = []
    y_next = []
    # Iterate in mini-batches for efficiency
    total = args.samples
    bs = max(1, int(args.batch_size))
    for start in range(0, total, bs):
        cur_bs = min(bs, total - start)
        ex = make_example(cfg, device=device, batch_size=cur_bs)
        # Compute neighbor counts for t for the whole batch
        from src.data.stream import ToroidalNeighborhood
        neigh = ToroidalNeighborhood(device=device)
        counts = neigh.neighbors(ex.t.to(torch.float32)).detach().cpu()  # [B,1,H,W]
        counts_np = counts[:, 0].reshape(cur_bs, HxW).numpy()  # [B, HxW]
        y_count.append(counts_np.reshape(-1))
        # Next state (ground truth t1) for the whole batch
        y_next.append(ex.t1.detach().cpu()[:, 0].reshape(cur_bs, HxW).numpy().reshape(-1))
        # Collect features for each layer
        for L in range(model.cfg.n_layers):
            x = collect_resid_pre(model, ex.tokens, ex.pos2d, L)  # [B,T,D] on CPU float32
            # Slice t segment only
            t_start = 1 + 18 + 1
            feats = x[:, t_start : t_start + HxW].reshape(cur_bs * HxW, -1).numpy()  # [B*HxW, D]
            X_by_layer.setdefault(L, []).append(feats)

    # Stack
    for L in X_by_layer:
        X_by_layer[L] = np.concatenate(X_by_layer[L], axis=0)  # [N*HxW, D]
    y_count = np.concatenate(y_count, axis=0)  # [N*HxW]
    y_next = np.concatenate(y_next, axis=0)  # [N*HxW]

    # Simple linear probes
    r2_by_layer = []
    auc_by_layer = []
    for L in range(model.cfg.n_layers):
        X = X_by_layer[L]
        # Count regression via least squares
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        w, *_ = np.linalg.lstsq(X1, y_count, rcond=None)
        pred = X1 @ w
        ss_res = ((pred - y_count) ** 2).sum()
        ss_tot = ((y_count - y_count.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        r2_by_layer.append(r2)
        # Next-alive logistic via linear scores
        # Fit via one-step Newton on least squares logits for simplicity
        y = y_next
        w2, *_ = np.linalg.lstsq(X1, y, rcond=None)
        logits = X1 @ w2
        # AUC
        order = np.argsort(-logits)
        yt = y[order]
        P = (yt == 1).sum(); N = (yt == 0).sum()
        if P > 0 and N > 0:
            tps = (yt == 1).cumsum(); fps = (yt == 0).cumsum()
            tpr = tps / P; fpr = fps / N
            auc = np.trapz(tpr, fpr)
        else:
            auc = np.nan
        auc_by_layer.append(auc)

    # Plot
    plt.figure(figsize=(5,3))
    plt.plot(r2_by_layer, marker='o'); plt.title('Neighbor-count R^2 by layer'); plt.xlabel('Layer'); plt.ylabel('R^2')
    plt.savefig(os.path.join(out, 'probe_count_r2.png'), dpi=150, bbox_inches='tight')
    plt.figure(figsize=(5,3))
    plt.plot(auc_by_layer, marker='o'); plt.title('Next-alive AUC by layer'); plt.xlabel('Layer'); plt.ylabel('AUC')
    plt.savefig(os.path.join(out, 'probe_next_auc.png'), dpi=150, bbox_inches='tight')
    print('saved:', out)


if __name__ == '__main__':
    main()
