"""2D RoPE for Q/K with row/col splits.

Expects q,k shaped (B, T, H, D). Applies rotary to first `rotary_dim` features:
- First half of `rotary_dim` uses row positions
- Second half uses col positions
Remainder dims (if any) untouched.
"""
from __future__ import annotations

import math
import torch


def _build_sin_cos(pos: torch.Tensor, dim: int, base: float = 10000.0, dtype: torch.dtype = torch.float32):
    """Build sin/cos for RoPE given positions [B,T] and half-dim (dim)."""
    device = pos.device
    # inv_freq: [dim//2]
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    angles = pos.to(torch.float32).unsqueeze(-1) * inv_freq  # [B,T,half]
    # Expand to pair-wise (even, odd) structure
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos.to(dtype), sin.to(dtype)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary to last dim of x using cos/sin with pair structure.

    x: [B,T,H,dim]
    cos/sin: [B,T,1,dim//2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    # Interleave back
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, pos2d: torch.Tensor, rotary_dim=None):
    B, T, H, D = q.shape
    assert k.shape == (B, T, H, D)
    assert pos2d.shape[:2] == (B, T) and pos2d.shape[-1] == 2
    dtype = q.dtype
    if rotary_dim is None:
        rotary_dim = D
    rotary_dim = int(rotary_dim)
    if rotary_dim % 4 != 0:
        # Need even for pairs and even split across row/col
        raise ValueError("rotary_dim must be divisible by 4 for 2D RoPE")
    half = rotary_dim // 2

    # Split q,k
    q_row, q_col, q_rest = q[..., :half], q[..., half:rotary_dim], q[..., rotary_dim:]
    k_row, k_col, k_rest = k[..., :half], k[..., half:rotary_dim], k[..., rotary_dim:]

    rows = pos2d[..., 0]
    cols = pos2d[..., 1]

    # Build cos/sin for row half and col half (pair structure)
    cos_r, sin_r = _build_sin_cos(rows, half, dtype=torch.float32)
    cos_c, sin_c = _build_sin_cos(cols, half, dtype=torch.float32)
    # Prepare for broadcasting over heads
    cos_r = cos_r.unsqueeze(2)  # [B,T,1,half/2]
    sin_r = sin_r.unsqueeze(2)
    cos_c = cos_c.unsqueeze(2)
    sin_c = sin_c.unsqueeze(2)

    # Apply rotary
    q_row_rot = _apply_rotary(q_row.to(torch.float32), cos_r, sin_r).to(dtype)
    k_row_rot = _apply_rotary(k_row.to(torch.float32), cos_r, sin_r).to(dtype)
    q_col_rot = _apply_rotary(q_col.to(torch.float32), cos_c, sin_c).to(dtype)
    k_col_rot = _apply_rotary(k_col.to(torch.float32), cos_c, sin_c).to(dtype)

    q_out = torch.cat([q_row_rot, q_col_rot, q_rest], dim=-1)
    k_out = torch.cat([k_row_rot, k_col_rot, k_rest], dim=-1)
    return q_out, k_out
