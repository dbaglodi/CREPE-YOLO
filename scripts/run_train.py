from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train import run_training
from training.utils import load_yaml, save_yaml, set_seed


def resolve_config_path(config_path: str) -> str:
    """Resolve config path relative to PROJECT_ROOT if it's relative."""
    p = Path(config_path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    out_cfg = Path(cfg["output_root"]) / cfg["run_name"] / "resolved_config.yaml"
    save_yaml(out_cfg, cfg)

    summary = run_training(cfg)
    print("Training complete.")
    print(summary)


if __name__ == "__main__":
    main()
