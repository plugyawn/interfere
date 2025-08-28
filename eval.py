from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="exp/life32", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Eval stub running with exp:", cfg.exp.name)


if __name__ == "__main__":
    main()

