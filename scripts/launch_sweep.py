from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="generated_sweeps")
    parser.add_argument("--submit", action="store_true", help="If set, launch each run immediately.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small starter sweep; expand later.
    learning_rates = [1e-3, 3e-4]
    batch_sizes = [4, 8]
    weight_decays = [0.0, 1e-4]

    generated = []

    for lr, bs, wd in itertools.product(learning_rates, batch_sizes, weight_decays):
        cfg = load_yaml(args.base_config)
        cfg["optim"]["lr"] = lr
        cfg["optim"]["weight_decay"] = wd
        cfg["data"]["batch_size"] = bs
        cfg["run_name"] = f"sweep_lr{lr}_bs{bs}_wd{wd}".replace(".", "p")

        config_path = out_dir / f"{cfg['run_name']}.yaml"
        save_yaml(config_path, cfg)
        generated.append(config_path)

        if args.submit:
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_train.py"), "--config", str(config_path)]
            print("Launching:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print(f"Generated {len(generated)} configs in {out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
