"""Simple plotting helpers for probes and patch maps."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def plot_probe_curves(probe_res: Dict[str, List[float]], savepath: str | None = None):
    for name, vals in probe_res.items():
        plt.plot(vals, label=name)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    return plt.gcf()
