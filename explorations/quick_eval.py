from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch

import sys
if __package__ is None or __package__ == "":
    # Ensure repo root is on sys.path when running as a file
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from explorations.utils import load_cfg, load_model, make_example
from src.train.loop import _acc_from_logits, _auc_from_logits
from src.model.rope2d import apply_rope_2d


def add_rope_hooks(model, pos2d):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="exp/life32")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
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

    total_batches = math.ceil(args.samples / args.batch_size)
    acc_list: List[float] = []
    auc_list: List[float] = []

    vocab = None

    for bi in range(total_batches):
        bs = min(args.batch_size, args.samples - bi * args.batch_size)
        ex = make_example(cfg, device=device, batch_size=bs)
        if vocab is None:
            vocab = ex.vocab
        hooks = add_rope_hooks(model, ex.pos2d)
        logits = model.run_with_hooks(ex.tokens, fwd_hooks=hooks)  # [B, T, V]
        acc = _acc_from_logits(logits, ex.tokens, ex.mask)
        auc = _auc_from_logits(logits, ex.tokens, ex.mask, vocab)
        acc_list.append(float(acc))
        auc_list.append(float(auc))

    mean_acc = sum(acc_list) / max(len(acc_list), 1)
    mean_auc = sum(auc_list) / max(len(auc_list), 1)
    print(f"Eval over {args.samples} samples (batch={args.batch_size}):")
    print(f"  Accuracy: {mean_acc:.4f}")
    print(f"  AUC:      {mean_auc:.4f}")


if __name__ == "__main__":
    main()
