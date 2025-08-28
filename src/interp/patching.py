"""Activation patching utilities using TransformerLens hooks."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def patch_cell_neighborhood(model, tokens_src, pos2d_src, tokens_cf, pos2d_cf, layer: int, center: Tuple[int, int], H: int, W: int):
    """Patch a 3x3 neighborhood around center (r,c) at given layer's resid_pre.

    Returns logits with patch applied.
    """
    r0, c0 = center
    # Compute indices for 3x3 neighborhood in the t segment
    start_t = 1 + 18 + 1
    idxs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = (r0 + dr) % H
            cc = (c0 + dc) % W
            idxs.append(start_t + rr * W + cc)

    idxs = torch.tensor(idxs, device=tokens_src.device)

    def _patch(x, hook):
        x2 = x.clone()
        x2[:, idxs, :] = hook.ctx["cf"][hook.name][:, idxs, :]
        return x2

    # Prepare cache from counterfactual
    _, cache_cf = model.run_with_cache(tokens_cf, return_type="logits")
    fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", _patch)]
    # Stash counterfactual activations in hook context
    def _pre(hook):
        hook.ctx["cf"] = cache_cf

    logits = model.run_with_hooks(tokens_src, return_type="logits", fwd_hooks=fwd_hooks, fwd_hooks_pre=[(f"blocks.{layer}.hook_resid_pre", _pre)])
    return logits

