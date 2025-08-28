from __future__ import annotations

import os
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="exp/life_smoke", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Hydra config loaded")
    print(OmegaConf.to_yaml(cfg))
    # S7 will call into src.train.loop.train_loop(cfg)
    stamp = datetime.now().isoformat()
    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"hydra_sanity_{stamp}.txt"), "w") as f:
        f.write("ok\n")
    print("Done")


if __name__ == "__main__":
    main()

