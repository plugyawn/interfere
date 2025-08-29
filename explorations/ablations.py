from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
if __package__ is None or __package__ == "":
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import load_cfg, load_model, make_example, out_dir
from src.model.rope2d import apply_rope_2d
from src.train.loop import _auc_from_logits


def build_rope_hooks(model, pos2d):
    rotary_dim = model.cfg.d_head

    def _hq(q, hook):
        return apply_rope_2d(q, q, pos2d.to(q.device), rotary_dim=rotary_dim)[0]

    def _hk(k, hook):
        return apply_rope_2d(k, k, pos2d.to(k.device), rotary_dim=rotary_dim)[0]

    hooks = []
    for L in range(model.cfg.n_layers):
        hooks.append((f"blocks.{L}.attn.hook_q", _hq))
        hooks.append((f"blocks.{L}.attn.hook_k", _hk))
    return hooks


def eval_auc(model, cfg, device, samples: int, batch_size: int, extra_hooks=None) -> float:
    total = samples
    bs = max(1, batch_size)
    auc_sum = 0.0
    n = 0
    for start in range(0, total, bs):
        cur_bs = min(bs, total - start)
        ex = make_example(cfg, device=device, batch_size=cur_bs)
        base_hooks = build_rope_hooks(model, ex.pos2d)
        all_hooks = base_hooks + (extra_hooks or [])
        logits = model.run_with_hooks(ex.tokens, fwd_hooks=all_hooks)
        auc = _auc_from_logits(logits, ex.tokens, ex.mask, ex.vocab)
        auc_sum += float(auc)
        n += 1
    return auc_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--layers", type=str, default="all", help="Comma list of layers to test or 'all'")
    args = ap.parse_args()

    cfg = load_cfg(args.config_name)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device=device, run_id=args.run_id, ckpt_path=args.ckpt)

    out = out_dir()

    # Baseline
    base_auc = eval_auc(model, cfg, device, args.samples, args.batch_size)

    # Determine layers to test
    if args.layers == "all":
        layer_list = list(range(model.cfg.n_layers))
    else:
        layer_list = [int(x) for x in args.layers.split(',') if x.strip()]

    drops = []  # (layer, head, delta_auc)
    for L in layer_list:
        for H in range(model.cfg.n_heads):
            scores_name = f"blocks.{L}.attn.hook_attn_scores"

            def make_ablate(L=L, H=H):
                def _ablate_scores(t, hook):
                    # t: [B, H, Q, K] — set this head's scores to large negative so softmax ~0
                    t[:, H, :, :] = -1e9
                    return t
                return _ablate_scores

            auc = eval_auc(
                model, cfg, device, args.samples, args.batch_size,
                extra_hooks=[(scores_name, make_ablate())]
            )
            drops.append((L, H, base_auc - auc))

    # Plot
    import numpy as np
    arr = np.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=np.float32)
    for L, H, d in drops:
        arr[L, H] = d
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(arr, aspect='auto', cmap='viridis')
    fig.colorbar(im, label='AUC drop (ablating head)')
    ax.set_xlabel('Head'); ax.set_ylabel('Layer'); ax.set_title('Head importance by AUC drop')
    # annotate top-k heads
    K = min(5, arr.size)
    flat = arr.reshape(-1)
    idxs = np.argsort(-flat)[:K]
    Hs = arr.shape[1]
    for i, idx in enumerate(idxs):
        L = idx // Hs; Hh = idx % Hs
        ax.text(Hh, L, f'{arr[L,Hh]:.2f}', color='white', ha='center', va='center')
    path = os.path.join(out, 'ablate_head_auc_drop.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('baseline AUC:', base_auc)
    print('saved:', path)


if __name__ == '__main__':
    main()
