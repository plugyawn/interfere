from __future__ import annotations

import argparse
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import (
    load_cfg,
    load_model,
    make_example,
    out_dir,
    build_trainlike_hooks,
    mask_targets_like_train,
)
from src.data.stream import get_default_vocab, ToroidalNeighborhood
from src.data.stream import _apply_rule as apply_rule_builtin


def collect_resid_pre(cfg, model, tokens, pos2d, layer: int):
    # Apply train-like hooks (2D RoPE on Q/K, attention mask, segment embeddings)
    # and capture resid_pre at the specified layer.
    name = f"blocks.{layer}.hook_resid_pre"
    box = {}

    def _cap(x, hook):
        # Move to CPU float32 for numpy compatibility
        box["x"] = x.detach().to(torch.float32).cpu()

    # Mask targets in inputs to match train-time masking
    tokens_in = mask_targets_like_train(tokens, vocab=get_default_vocab(), cfg=cfg)
    hooks = build_trainlike_hooks(cfg, model, pos2d, tokens_in)
    hooks = [*hooks, (name, _cap)]
    model.run_with_hooks(tokens_in, return_type=None, fwd_hooks=hooks)
    return box["x"]  # [B, T, D]


def collect_resid_activation(cfg, model, tokens, pos2d, layer: int, kind: str):
    """Capture a specific residual/activation at a given layer with train-like hooks.

    kind in {"resid_pre","resid_post","attn_out","mlp_out"}
    """
    hook_name = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "attn_out": f"blocks.{layer}.hook_attn_out",
        "mlp_out": f"blocks.{layer}.hook_mlp_out",
    }.get(kind)
    if hook_name is None:
        raise ValueError(f"Unknown kind: {kind}")
    box = {}

    def _cap(x, hook):
        box["x"] = x.detach().to(torch.float32).cpu()

    tokens_in = mask_targets_like_train(tokens, vocab=get_default_vocab(), cfg=cfg)
    hooks = build_trainlike_hooks(cfg, model, pos2d, tokens_in)
    hooks = [*hooks, (hook_name, _cap)]
    model.run_with_hooks(tokens_in, return_type=None, fwd_hooks=hooks)
    return box["x"]


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    yt = labels[order]
    P = (yt == 1).sum()
    N = (yt == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    tps = (yt == 1).cumsum(); fps = (yt == 0).cumsum()
    tpr = tps / P; fpr = fps / N
    auc = np.trapz(tpr, fpr)
    return float(auc)


def _fit_linear_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return X1 @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--features", default="resid_pre", help="Comma-separated: resid_pre,resid_post,attn_out,mlp_out (currently uses first)")
    ap.add_argument("--sweep", action="store_true", help="Run multiple targets and save JSON summaries + plots")
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
    out = out_dir()

    # Build a small dataset
    X_by_layer = {}
    y_count = []
    y_next = []
    y_curr = []
    # neighbor bits (8) and center bit
    nb_bits = [list() for _ in range(8)]
    y_center = []
    # count bin labels eq k (0..8)
    y_count_eq = [list() for _ in range(9)]
    # thresholds
    y_ge2, y_ge3, y_le1 = [], [], []
    # coords for t-segment
    rows_flat = np.repeat(np.arange(H), W)
    cols_flat = np.tile(np.arange(W), H)
    y_row, y_col = [], []
    # Iterate in mini-batches for efficiency
    total = args.samples
    bs = max(1, int(args.batch_size))
    for start in range(0, total, bs):
        cur_bs = min(bs, total - start)
        ex = make_example(cfg, device=device, batch_size=cur_bs)
        # Compute neighbor counts for t for the whole batch
        neigh = ToroidalNeighborhood(device=device)
        counts = neigh.neighbors(ex.t.to(torch.float32)).detach().cpu()  # [B,1,H,W]
        counts_np = counts[:, 0].reshape(cur_bs, HxW).numpy()  # [B, HxW]
        y_count.append(counts_np.reshape(-1))
        # Next state (ground truth t1) for the whole batch
        y_next.append(ex.t1.detach().cpu()[:, 0].reshape(cur_bs, HxW).numpy().reshape(-1))
        # Current state (ground truth t)
        t_np = ex.t.detach().cpu()[:, 0].reshape(cur_bs, HxW).numpy().reshape(-1)
        y_curr.append(t_np)
        # center bit
        y_center.append(t_np)
        # neighbor bits via toroidal shifts on GPU then CPU
        t_b = ex.t  # [B,1,H,W]
        shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for idx, (dr, dc) in enumerate(shifts):
            nb = torch.roll(t_b, shifts=(dr, dc), dims=(-2, -1))
            nb_np = nb.detach().cpu()[:, 0].reshape(cur_bs, HxW).numpy().reshape(-1)
            nb_bits[idx].append(nb_np)
        # count bins and thresholds
        for k in range(9):
            y_count_eq[k].append((counts_np.reshape(-1) == k).astype(np.int64))
        y_ge2.append((counts_np.reshape(-1) >= 2).astype(np.int64))
        y_ge3.append((counts_np.reshape(-1) >= 3).astype(np.int64))
        y_le1.append((counts_np.reshape(-1) <= 1).astype(np.int64))
        # coords
        y_row.append(np.tile(rows_flat, cur_bs))
        y_col.append(np.tile(cols_flat, cur_bs))
        # Collect features for each layer
        for L in range(model.cfg.n_layers):
            # For now, only use the first requested feature kind (default resid_pre)
            feat_kind = args.features.split(',')[0]
            if feat_kind == 'resid_pre':
                x = collect_resid_pre(cfg, model, ex.tokens, ex.pos2d, L)
            else:
                x = collect_resid_activation(cfg, model, ex.tokens, ex.pos2d, L, feat_kind)
            # Slice t segment only
            t_start = 1 + 18 + 1
            feats = x[:, t_start : t_start + HxW].reshape(cur_bs * HxW, -1).numpy()  # [B*HxW, D]
            X_by_layer.setdefault(L, []).append(feats)

    # Stack
    for L in X_by_layer:
        X_by_layer[L] = np.concatenate(X_by_layer[L], axis=0)  # [N*HxW, D]
    y_count = np.concatenate(y_count, axis=0)  # [N*HxW]
    y_next = np.concatenate(y_next, axis=0)  # [N*HxW]
    y_curr = np.concatenate(y_curr, axis=0)  # [N*HxW]
    y_center = np.concatenate(y_center, axis=0)
    nb_bits = [np.concatenate(v, axis=0) for v in nb_bits]
    y_count_eq = [np.concatenate(v, axis=0) for v in y_count_eq]
    y_ge2 = np.concatenate(y_ge2, axis=0)
    y_ge3 = np.concatenate(y_ge3, axis=0)
    y_le1 = np.concatenate(y_le1, axis=0)
    y_row = np.concatenate(y_row, axis=0).astype(np.float32)
    y_col = np.concatenate(y_col, axis=0).astype(np.float32)

    # Simple linear probes
    r2_by_layer = []              # count regression R^2
    auc_by_layer = []             # next-alive AUC
    curr_auc_by_layer = []        # curr-alive AUC
    strat_auc_alive0 = []         # next AUC | curr=0
    strat_auc_alive1 = []         # next AUC | curr=1
    nb_mean_auc = []              # mean neighbor-bit AUC over 8 neighbors
    center_auc = []               # center bit AUC
    eqk_mean_auc = []             # mean over k=0..8 one-vs-rest
    ge_le_aucs = {"ge2": [], "ge3": [], "le1": []}
    row_r2 = []
    col_r2 = []
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
        # Next-alive AUC from linear scores
        logits = _fit_linear_scores(X, y_next)
        auc_by_layer.append(_auc_from_scores(logits, y_next))

        # Current-alive probe (should be easier than next-alive)
        logits_c = _fit_linear_scores(X, y_curr)
        curr_auc_by_layer.append(_auc_from_scores(logits_c, y_curr))

        # Stratified next-alive AUCs
        m0 = (y_curr == 0)
        m1 = (y_curr == 1)
        if m0.any():
            strat_auc_alive0.append(_auc_from_scores(logits[m0], y_next[m0]))
        else:
            strat_auc_alive0.append(float('nan'))
        if m1.any():
            strat_auc_alive1.append(_auc_from_scores(logits[m1], y_next[m1]))
        else:
            strat_auc_alive1.append(float('nan'))

        # Neighbor bits AUC (8 neighbors) and center bit
        nb_aucs = []
        for i in range(8):
            logits_nb = _fit_linear_scores(X, nb_bits[i])
            nb_aucs.append(_auc_from_scores(logits_nb, nb_bits[i]))
        nb_mean_auc.append(float(np.nanmean(nb_aucs)))
        logits_center = _fit_linear_scores(X, y_center)
        center_auc.append(_auc_from_scores(logits_center, y_center))

        # Count eq k one-vs-rest (mean AUC over k)
        eq_aucs = []
        for k in range(9):
            logits_k = _fit_linear_scores(X, y_count_eq[k])
            eq_aucs.append(_auc_from_scores(logits_k, y_count_eq[k]))
        eqk_mean_auc.append(float(np.nanmean(eq_aucs)))

        # Thresholds AUCs
        for name, yb in (('ge2', y_ge2), ('ge3', y_ge3), ('le1', y_le1)):
            logits_b = _fit_linear_scores(X, yb)
            ge_le_aucs[name].append(_auc_from_scores(logits_b, yb))

        # Row/col regression R^2
        for tgt, store in ((y_row, row_r2), (y_col, col_r2)):
            w4, *_ = np.linalg.lstsq(X1, tgt, rcond=None)
            pred_xy = X1 @ w4
            ss_res_xy = ((pred_xy - tgt) ** 2).sum()
            ss_tot_xy = ((tgt - tgt.mean()) ** 2).sum()
            store.append(1.0 - ss_res_xy / max(ss_tot_xy, 1e-9))

    # Plot core probes
    plt.figure(figsize=(5,3))
    plt.plot(r2_by_layer, marker='o'); plt.title('Neighbor-count R^2 by layer'); plt.xlabel('Layer'); plt.ylabel('R^2')
    plt.savefig(os.path.join(out, 'probe_count_r2.png'), dpi=150, bbox_inches='tight')
    plt.figure(figsize=(5,3))
    plt.plot(auc_by_layer, marker='o'); plt.title('Next-alive AUC by layer'); plt.xlabel('Layer'); plt.ylabel('AUC')
    plt.savefig(os.path.join(out, 'probe_next_auc.png'), dpi=150, bbox_inches='tight')
    plt.figure(figsize=(5,3))
    plt.plot(curr_auc_by_layer, marker='o'); plt.title('Current-alive AUC by layer'); plt.xlabel('Layer'); plt.ylabel('AUC')
    plt.savefig(os.path.join(out, 'probe_curr_auc.png'), dpi=150, bbox_inches='tight')
    # Stratified next-alive
    plt.figure(figsize=(5,3))
    plt.plot(strat_auc_alive0, marker='o', label='alive=0'); plt.plot(strat_auc_alive1, marker='s', label='alive=1')
    plt.title('Next-alive AUC (stratified)'); plt.xlabel('Layer'); plt.ylabel('AUC'); plt.legend()
    plt.savefig(os.path.join(out, 'probe_next_auc_stratified.png'), dpi=150, bbox_inches='tight')
    # Neighbor bits mean AUC and center
    plt.figure(figsize=(5,3))
    plt.plot(nb_mean_auc, marker='o', label='neighbors mean'); plt.plot(center_auc, marker='s', label='center')
    plt.title('Bit AUC (neighbors mean + center)'); plt.xlabel('Layer'); plt.ylabel('AUC'); plt.legend()
    plt.savefig(os.path.join(out, 'probe_bits_auc.png'), dpi=150, bbox_inches='tight')
    # Count bins mean AUC and thresholds
    plt.figure(figsize=(5,3))
    plt.plot(eqk_mean_auc, marker='o', label='count eq k (mean)')
    plt.plot(ge_le_aucs['ge2'], marker='^', label='count≥2')
    plt.plot(ge_le_aucs['ge3'], marker='v', label='count≥3')
    plt.plot(ge_le_aucs['le1'], marker='s', label='count≤1')
    plt.title('Count bins/threshold AUC'); plt.xlabel('Layer'); plt.ylabel('AUC'); plt.legend()
    plt.savefig(os.path.join(out, 'probe_count_bins_auc.png'), dpi=150, bbox_inches='tight')
    # Row/col R2
    plt.figure(figsize=(5,3))
    plt.plot(row_r2, marker='o', label='row R^2'); plt.plot(col_r2, marker='s', label='col R^2')
    plt.title('Coords R^2'); plt.xlabel('Layer'); plt.ylabel('R^2'); plt.legend()
    plt.savefig(os.path.join(out, 'probe_coords_r2.png'), dpi=150, bbox_inches='tight')
    # Print quick summary
    print('Count R^2 by layer:', np.round(r2_by_layer, 4))
    print('Next-alive AUC by layer:', np.round(auc_by_layer, 4))
    print('Current-alive AUC by layer:', np.round(curr_auc_by_layer, 4))
    print('Next-alive stratified AUC (alive=0):', np.round(strat_auc_alive0, 4))
    print('Next-alive stratified AUC (alive=1):', np.round(strat_auc_alive1, 4))
    print('Neighbor bits mean AUC:', np.round(nb_mean_auc, 4))
    print('Center bit AUC:', np.round(center_auc, 4))
    print('Count eq k mean AUC:', np.round(eqk_mean_auc, 4))
    print('Threshold AUCs ge2:', np.round(ge_le_aucs['ge2'], 4))
    print('Threshold AUCs ge3:', np.round(ge_le_aucs['ge3'], 4))
    print('Threshold AUCs le1:', np.round(ge_le_aucs['le1'], 4))
    print('Row/Col R^2:', np.round(row_r2, 4), np.round(col_r2, 4))

    if args.sweep:
        summary = {
            'count_r2': r2_by_layer,
            'next_auc': auc_by_layer,
            'curr_auc': curr_auc_by_layer,
            'next_auc_alive0': strat_auc_alive0,
            'next_auc_alive1': strat_auc_alive1,
            'neighbor_bits_mean_auc': nb_mean_auc,
            'center_auc': center_auc,
            'count_eqk_mean_auc': eqk_mean_auc,
            'thresh_auc': ge_le_aucs,
            'row_r2': row_r2,
            'col_r2': col_r2,
            'layers': model.cfg.n_layers,
            'samples': args.samples,
            'batch_size': args.batch_size,
            'features': args.features.split(',')[0],
        }
        with open(os.path.join(out, 'probe_sweep_summary.json'), 'w') as f:
            json.dump(summary, f)
    print('saved:', out)


if __name__ == '__main__':
    main()
