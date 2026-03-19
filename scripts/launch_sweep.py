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


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="generated_sweeps")
    parser.add_argument("--submit", action="store_true", help="Submit the generated sweep as a SLURM array.")
    parser.add_argument("--array-limit", type=int, default=8, help="Maximum concurrent array tasks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = resolve_config_path(args.base_config)
    out_dir = resolve_config_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(str(base_config_path))

    learning_rates = [1e-3, 5e-4, 3e-4]
    batch_sizes = [4, 8, 16]
    weight_decays = [0.0, 5e-5, 1e-4]
    grad_clips = [3.0, 5.0]

    generated = []

    for lr, bs, wd, gc in itertools.product(learning_rates, batch_sizes, weight_decays, grad_clips):
        cfg = load_yaml(str(base_config_path))
        cfg["optim"]["lr"] = lr
        cfg["optim"]["weight_decay"] = wd
        cfg["data"]["batch_size"] = bs
        cfg["train"]["grad_clip"] = gc
        cfg["run_name"] = f"sweep_lr{lr}_bs{bs}_wd{wd}_gc{gc}".replace(".", "p")

        config_path = out_dir / f"{cfg['run_name']}.yaml"
        save_yaml(str(config_path), cfg)
        generated.append(config_path)

    manifest_path = out_dir / "configs.txt"
    manifest_path.write_text("\n".join(str(path) for path in generated) + "\n")

    print(f"Generated {len(generated)} configs in {out_dir}")
    print(f"Manifest: {manifest_path}")
    for path in generated:
        print(path)

    if args.submit:
        array_spec = f"0-{len(generated) - 1}%{args.array_limit}"
        cmd = [
            "sbatch",
            f"--array={array_spec}",
            str(PROJECT_ROOT / "scripts" / "slurm_sweep.sh"),
            str(manifest_path),
        ]
        print("Submitting:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
