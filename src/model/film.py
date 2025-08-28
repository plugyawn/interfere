"""FiLM conditioner placeholder for S1.

S10 will add implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RuleFiLM(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

    def forward(self, rule_bits: torch.Tensor):
        raise NotImplementedError

