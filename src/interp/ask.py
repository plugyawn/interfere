"""Ask-Me-Anything API.

Example calls:
  answer("which_heads_count_neighbors", rule="S23/B3", layer=3)
  answer("linear_probe", feature="neighbor_k", layer=5)
  answer("patch_cell", coords=(10,12), layer=6, patch="neighbor=3")
"""
from __future__ import annotations

from typing import Any, Dict

from .probes import run_probes


def answer(query: str, **kwargs) -> Dict[str, Any]:
    if query == "linear_probe":
        model = kwargs["model"]
        cfg = kwargs["cfg"]
        return run_probes(model, cfg)
    raise NotImplementedError(f"Unknown query: {query}")
