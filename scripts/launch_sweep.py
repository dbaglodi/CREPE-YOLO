from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import subprocess
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / config_path


def load_yaml(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(config_path: Path, config: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="generated_sweeps")
    parser.add_argument("--submit", action="store_true", help="Submit the generated sweep as a SLURM array.")
    parser.add_argument("--array-limit", type=int, default=8, help="Maximum concurrent array tasks.")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 3e-4],
        help="Learning rates to sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Batch sizes to sweep. Defaults avoid the bs16 OOMs seen on the current setup.",
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=[0.0, 5e-5, 1e-4],
        help="Weight decays to sweep.",
    )
    parser.add_argument(
        "--grad-clips",
        type=float,
        nargs="+",
        default=[3.0, 5.0],
        help="Gradient clipping values to sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = resolve_config_path(args.base_config)
    out_dir = resolve_config_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_config_path)

    learning_rates = args.learning_rates
    batch_sizes = args.batch_sizes
    weight_decays = args.weight_decays
    grad_clips = args.grad_clips

    generated = []

    for lr, bs, wd, gc in itertools.product(learning_rates, batch_sizes, weight_decays, grad_clips):
        cfg = load_yaml(base_config_path)
        cfg["optim"]["lr"] = lr
        cfg["optim"]["weight_decay"] = wd
        cfg["data"]["batch_size"] = bs
        cfg["train"]["grad_clip"] = gc
        cfg["run_name"] = f"sweep_lr{lr}_bs{bs}_wd{wd}_gc{gc}".replace(".", "p")

        config_path = out_dir / f"{cfg['run_name']}.yaml"
        save_yaml(config_path, cfg)
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
