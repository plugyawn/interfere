"""Linear probes for neighbor-k, prev-alive, and decision.

S11: lightweight implementation optimized for small batches.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.rules import sb_to_bits
from ..data.stream import ToroidalNeighborhood, assemble_sequence, get_default_vocab, make_batch


def _collect_resids(model, tokens, pos2d, hooks_names: List[str]):
    # Use cache to grab residual streams at desired points
    # names like 'blocks.{l}.hook_resid_pre'
    logits, cache = model.run_with_cache(tokens, return_type="logits", names_filter=lambda n: any(h in n for h in hooks_names))
    resids = []
    for name in [n for n in cache.keys() if any(h in n for h in hooks_names)]:
        resids.append(cache[name])
    return logits, resids, cache


def linear_probe_neighbor_k(model, cfg, batches: int = 4) -> Dict[str, List[float]]:
    device = next(model.parameters()).device
    H, W = cfg.board.H, cfg.board.W
    vocab = get_default_vocab()
    rule_bits = sb_to_bits({2, 3}, {3}).to(device)
    neigh = ToroidalNeighborhood(device=device)

    feats_by_layer: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for _ in range(batches):
        rb, t, t1 = make_batch(B=8, H=H, W=W, device=device, rules=rule_bits)
        tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab)
        # Collect resid_pre per layer
        hook_names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
        logits, resids, _ = _collect_resids(model, tokens, pos2d, hook_names)
        # Get neighbor counts for prefix board tokens
        counts = neigh.neighbors(t.to(torch.float32)).squeeze(1).long()  # [B,H,W]
        y = counts.view(-1)  # labels 0..8
        labels_all.append(y)
        # Extract features at t segment
        start_t = 1 + 18 + 1
        fspan = slice(start_t, start_t + H * W)
        for l, r in enumerate(resids):
            r2 = r[:, fspan, :].reshape(-1, r.size(-1))  # [B*H*W,D]
            if len(feats_by_layer) <= l:
                feats_by_layer.append(r2.detach())
            else:
                feats_by_layer[l] = torch.cat([feats_by_layer[l], r2.detach()], dim=0)

    Y = torch.cat(labels_all, dim=0)
    accs: List[float] = []
    for l, X in enumerate(feats_by_layer):
        clf = nn.Linear(X.size(1), 9).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=0.0)
        for _ in range(200):
            logits = clf(X)
            loss = F.cross_entropy(logits, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = clf(X).argmax(dim=-1)
            acc = (pred == Y).float().mean().item()
        accs.append(acc)
    return {"neighbor_k": accs}


def run_probes(model, cfg) -> Dict[str, List[float]]:
    res = linear_probe_neighbor_k(model, cfg)
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"probes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    return res

