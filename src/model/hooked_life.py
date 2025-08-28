"""HookedTransformer wrapper placeholder for S1.

S6 will construct a TransformerLens model and add 2D RoPE hooks.
"""
from __future__ import annotations

from typing import Any


def build_model(cfg: Any):
    raise NotImplementedError("Implemented in S6")

