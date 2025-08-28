"""Rule encoding/decoding (S/B) stubs for S1.

Implemented fully in S2. For now, provide minimal placeholders so imports work.
"""
from __future__ import annotations

from typing import Set, Tuple

import torch


def sb_to_bits(s_set: Set[int], b_set: Set[int]) -> torch.BoolTensor:
    """Placeholder: returns an 18-length False tensor.

    S2 will implement actual encoding.
    """
    bits = torch.zeros(18, dtype=torch.bool)
    return bits


def bits_to_sb(bits: torch.Tensor) -> Tuple[Set[int], Set[int]]:
    """Placeholder: returns empty S,B sets.

    S2 will implement actual decoding.
    """
    return set(), set()


def parse_rule(s: str) -> Tuple[Set[int], Set[int]]:
    """Placeholder parser: accepts strings like 'S23/B3' in S2.
    Returns empty sets for now.
    """
    return set(), set()


PREDEFINED = {
    "S23/B3": (set(), set()),
    "S23/B36": (set(), set()),
    "B2/S": (set(), set()),
}

