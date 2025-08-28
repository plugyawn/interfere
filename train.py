from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.train.loop import train_loop, train_loop_ddp


@hydra.main(config_path="cfg", config_name="exp/life_smoke", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Config:\n" + OmegaConf.to_yaml(cfg))
    import os
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        stats = train_loop_ddp(cfg)
    else:
        stats = train_loop(cfg)
    print(
        {
            "loss": round(stats.loss, 6),
            "acc": round(stats.acc, 6),
            "tokens_per_step": stats.tokens_per_step,
            "tps": stats.tps,
            "secs": stats.secs,
        }
    )


if __name__ == "__main__":
    main()
