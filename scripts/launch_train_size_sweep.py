from __future__ import annotations

import argparse
import glob
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
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Resolved config for the selected hyperparameters.",
    )
    parser.add_argument(
        "--logs-glob",
        type=str,
        default=None,
        help="Optional sweep log glob. If provided, select the best tuning config automatically.",
    )
    parser.add_argument("--output-dir", type=str, default="generated_sweeps/train_size")
    parser.add_argument("--submit", action="store_true", help="Submit the generated sweep as a SLURM array.")
    parser.add_argument("--array-limit", type=int, default=8, help="Maximum concurrent array tasks.")
    parser.add_argument(
        "--train-set-usages",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Fractions of the train+val pool to use for retraining.",
    )
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def parse_best_config_from_logs(logs_glob: str) -> Path:
    from scripts.summarize_sweep import parse_log

    paths = [Path(path) for path in sorted(glob.glob(logs_glob))]
    if not paths:
        raise SystemExit(f"No logs matched: {logs_glob}")

    rows = [parse_log(path) for path in paths]
    ranked = sorted(
        [row for row in rows if row["best_val_f1_op"] is not None],
        key=lambda row: row["best_val_f1_op"],
        reverse=True,
    )
    if not ranked:
        raise SystemExit(f"No completed tuning runs with validation metrics matched: {logs_glob}")

    best = ranked[0]
    resolved_config_path = best.get("resolved_config_path")
    if not resolved_config_path:
        raise SystemExit(f"Best run {best['config']} did not expose a resolved config path.")

    path = Path(resolved_config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise SystemExit(f"Resolved config not found: {path}")

    print(
        "Selected best tuning config from logs:",
        best["config"],
        f"(best_val_F1_op={best['best_val_f1_op']:.4f})",
    )
    print(f"Resolved config: {path}")
    return path


def main() -> None:
    args = parse_args()
    if bool(args.base_config) == bool(args.logs_glob):
        raise SystemExit("Provide exactly one of --base-config or --logs-glob.")

    if args.logs_glob:
        base_config_path = parse_best_config_from_logs(args.logs_glob)
    else:
        base_config_path = resolve_config_path(args.base_config)

    out_dir = resolve_config_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for usage in args.train_set_usages:
        cfg = load_yaml(base_config_path)
        cfg.setdefault("data", {})
        cfg.setdefault("train", {})
        cfg["data"]["combine_val_to_train"] = True
        cfg["data"]["train_set_usage"] = usage
        cfg["train"]["save_every"] = args.save_every
        run_name = cfg.get("run_name", "selected_config")
        usage_tag = str(usage).replace(".", "p")
        cfg["run_name"] = f"{run_name}_usage{usage_tag}"

        config_path = out_dir / f"{cfg['run_name']}.yaml"
        save_yaml(config_path, cfg)
        generated.append(config_path)

    manifest_path = out_dir / "configs.txt"
    manifest_path.write_text("\n".join(str(path) for path in generated) + "\n", encoding="utf-8")

    print(f"Generated {len(generated)} train-size configs in {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Base config: {base_config_path}")
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
