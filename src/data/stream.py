"""On-GPU data stream stubs for S1.

S3 will implement toroidal conv2d neighbor counts and batching.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class ToroidalNeighborhood:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def neighbors(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder: returns zeros with same shape
        return torch.zeros_like(x)


def make_batch(
    B: int,
    H: int,
    W: int,
    density_mix: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = None,
    rules: Optional[torch.Tensor] = None,
    structured_prob: float = 0.1,
):
    """Placeholder batch generator. Returns empty tensors of correct shapes.

    S3 will implement real sampling and transitions.
    """
    rb = torch.zeros((B, 18), dtype=torch.bool, device=device)
    t = torch.zeros((B, 1, H, W), dtype=torch.long, device=device)
    t1 = torch.zeros_like(t)
    return rb, t, t1


def assemble_sequence(rule_bits, t, t1, vocab=None):
    """S4 will implement tokenization; placeholder returns minimal tensors."""
    B, _, H, W = t.shape
    T = 1 + 18 + 1 + H * W + 1 + H * W
    tokens = torch.zeros((B, T), dtype=torch.long, device=t.device)
    loss_mask = torch.zeros((B, T), dtype=torch.bool, device=t.device)
    pos2d = torch.zeros((B, T, 2), dtype=torch.long, device=t.device)
    return tokens, loss_mask, pos2d

