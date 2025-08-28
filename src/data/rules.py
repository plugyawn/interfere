"""Rule encoding/decoding (S/B) utilities.

Encoding: 18 bits [S0..S8, B0..B8] as torch.bool.
"""
from __future__ import annotations

from typing import Iterable, Set, Tuple

import torch


def _validate_counts(counts: Iterable[int]) -> None:
    for c in counts:
        if not (0 <= int(c) <= 8):
            raise ValueError(f"Neighbor count out of range [0,8]: {c}")


def sb_to_bits(s_set: Set[int], b_set: Set[int]) -> torch.BoolTensor:
    """Encode survival/birth sets into 18-bit bool tensor [S0..S8,B0..B8]."""
    _validate_counts(s_set)
    _validate_counts(b_set)
    bits = torch.zeros(18, dtype=torch.bool)
    for c in s_set:
        bits[c] = True
    for c in b_set:
        bits[9 + c] = True
    return bits


def bits_to_sb(bits: torch.Tensor) -> Tuple[Set[int], Set[int]]:
    """Decode 18-bit bool/int tensor to (S_set, B_set)."""
    if bits.numel() != 18:
        raise ValueError(f"Expected 18 bits, got shape {tuple(bits.shape)}")
    b = bits.to(torch.bool).flatten()
    s_set = {i for i in range(9) if bool(b[i].item())}
    b_set = {i for i in range(9) if bool(b[9 + i].item())}
    return s_set, b_set


def parse_rule(s: str) -> Tuple[Set[int], Set[int]]:
    """Parse strings like 'S23/B3' or 'B3/S23' or 'B2/S'."""
    s = s.strip()
    if "/" not in s:
        raise ValueError("Rule must contain '/' separator, e.g., 'S23/B3'")
    left, right = s.split("/", 1)
    left = left.strip()
    right = right.strip()

    def parse_part(part: str) -> Tuple[str, Set[int]]:
        if not part:
            return "", set()
        part = part.upper()
        if part[0] not in ("S", "B"):
            raise ValueError(f"Invalid rule part '{part}', must start with 'S' or 'B'")
        kind = part[0]
        digits = part[1:]
        counts: Set[int] = set()
        for ch in digits:
            if not ch:
                continue
            if not ch.isdigit():
                raise ValueError(f"Invalid digit '{ch}' in rule '{part}'")
            val = int(ch)
            if val < 0 or val > 8:
                raise ValueError(f"Neighbor count out of range [0,8]: {val}")
            counts.add(val)
        return kind, counts

    k1, c1 = parse_part(left)
    k2, c2 = parse_part(right)

    if k1 == "S" and k2 == "B":
        return c1, c2
    if k1 == "B" and k2 == "S":
        return c2, c1
    raise ValueError(f"Malformed rule string: '{s}'")


def sb_to_string(s_set: Set[int], b_set: Set[int]) -> str:
    s_digits = "".join(str(d) for d in sorted(s_set))
    b_digits = "".join(str(d) for d in sorted(b_set))
    return f"S{s_digits}/B{b_digits}"


def bits_to_string(bits: torch.Tensor) -> str:
    s_set, b_set = bits_to_sb(bits)
    return sb_to_string(s_set, b_set)


def random_rule_bits(p: float = 0.5, device: torch.device | None = None) -> torch.BoolTensor:
    """Sample a random rule with independent Bernoulli(p) bits."""
    return (torch.rand(18, device=device) < p)


PREDEFINED = {
    "S23/B3": ({2, 3}, {3}),
    "S23/B36": ({2, 3}, {3, 6}),
    "B2/S": (set(), {2}),
}
