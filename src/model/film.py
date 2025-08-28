"""FiLM conditioner mapping 18-bit rule -> per-block (gamma, beta)."""
from __future__ import annotations

import torch
import torch.nn as nn


class RuleFiLM(nn.Module):
    def __init__(self, n_layers: int, d_model: int, hidden: int = 256, zero_init: bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_layers * d_model),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, rule_bits: torch.Tensor):
        """Return gamma,beta shaped [B, n_layers, d_model].

        We parameterize residual as: resid = resid * (1 + gamma) + beta
        """
        x = rule_bits.to(torch.float32)
        out = self.net(x)  # [B, 2*n_layers*d_model]
        out = out.view(x.size(0), 2, self.n_layers, self.d_model)
        gamma, beta = out[:, 0], out[:, 1]
        return gamma, beta
