from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="exp/life32", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Interp run stub. Action TBD.")


if __name__ == "__main__":
    main()

