"""2D RoPE placeholder for S1.

S5 will implement proper 2D rotary embeddings for Q/K.
"""
from __future__ import annotations

import torch


def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, pos2d: torch.Tensor, rotary_dim=None):
    # Placeholder: no-op pass-through
    return q, k

