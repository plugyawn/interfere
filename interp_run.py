from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from omegaconf import DictConfig

from src.interp.probes import run_probes
from src.model.hooked_life import build_model


@hydra.main(config_path="cfg", config_name="exp/life32", version_base=None)
def main(cfg: DictConfig) -> None:
    action = cfg.get("action", "probes")
    model, _ = build_model(cfg)
    if action == "probes":
        res = run_probes(model, cfg)
        print(json.dumps(res))
    elif action == "patch_maps":
        # Placeholder for patch maps generation; could be extended
        print("Patch maps not yet implemented")
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
