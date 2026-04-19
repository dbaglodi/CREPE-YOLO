from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(r"End of Epoch (\d+) \| (.*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CREPE-YOLO sweep logs by best validation F1(op)."
    )
    parser.add_argument(
        "--logs-glob",
        type=str,
        required=True,
        help="Glob for sweep log files, for example 'logs/crepe_sweep_a40_4501620_*.out'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top runs to print.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_")


def coerce_float(value: str):
    if value == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_usage(config_name: str):
    match = re.search(r"_usage(\d+p\d+)", config_name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def fmt_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def parse_log(path: Path) -> dict:
    config = None
    status = "unknown"
    checkpoint_dir = None
    epochs = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"Running sweep task \d+ with config: .*/(sweep_[^/]+)\.ya?ml", line)
        if match:
            config = match.group(1)
        else:
            match = re.search(r"Running sweep task \d+ with config: .*/([^/]+)\.ya?ml", line)
            if match:
                config = match.group(1)

        match = re.search(r"Checkpoint directory: (.+)", line)
        if match:
            checkpoint_dir = match.group(1).strip()

        match = EPOCH_PATTERN.search(line)
        if match:
            epoch = int(match.group(1))
            metrics = {"epoch": epoch}
            for label, value in re.findall(r"([^:|]+): ([0-9.]+|nan)", match.group(2)):
                metrics[normalize_label(label)] = coerce_float(value)
            epochs.append(metrics)

        if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
            status = "oom"
        elif "--- Training Complete ---" in line and status != "oom":
            status = "completed"

    best_epoch_metrics = None
    best_val_f1 = None
    best_test_f1 = None
    if epochs:
        candidates = [epoch for epoch in epochs if epoch.get("val_f1_op") is not None]
        if candidates:
            best_epoch_metrics = max(candidates, key=lambda epoch: epoch["val_f1_op"])
            best_val_f1 = best_epoch_metrics["val_f1_op"]
        else:
            candidates = [epoch for epoch in epochs if epoch.get("test_f1_op") is not None]
            if candidates:
                best_epoch_metrics = max(candidates, key=lambda epoch: epoch["test_f1_op"])
        if best_epoch_metrics is not None:
            best_test_f1 = best_epoch_metrics.get("test_f1_op")

    last_metrics = epochs[-1] if epochs else {}
    best_checkpoint_path = None
    if checkpoint_dir and best_epoch_metrics is not None:
        if best_val_f1 is not None:
            best_checkpoint_path = str(Path(checkpoint_dir) / "best_val_f1_op.pt")
        elif best_test_f1 is not None:
            best_checkpoint_path = str(Path(checkpoint_dir) / "best_test_f1_op.pt")

    return {
        "log_path": str(path),
        "config": config or path.stem,
        "status": status,
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint_path": best_checkpoint_path,
                "resolved_config_path": (
            str(Path(checkpoint_dir).parent / "resolved_config.yaml")
            if checkpoint_dir
            else None
        ),
        "best_epoch": best_epoch_metrics["epoch"] if best_epoch_metrics else None,
        "best_val_f1_op": best_val_f1,
        "test_f1_op_at_best_val": best_test_f1,
        "last_epoch": last_metrics.get("epoch"),
        "last_val_f1_op": last_metrics.get("val_f1_op"),
        "last_test_f1_op": last_metrics.get("test_f1_op"),
        "train_set_usage": extract_usage(config or path.stem),
        "epochs": epochs,
    }


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.logs_glob))]
    if not paths:
        raise SystemExit(f"No logs matched: {args.logs_glob}")

    rows = [parse_log(path) for path in paths]
    ranked = sorted(
        [row for row in rows if row["best_val_f1_op"] is not None],
        key=lambda row: row["best_val_f1_op"],
        reverse=True,
    )
    utilization_rows = sorted(
        [row for row in rows if row["train_set_usage"] is not None and row["last_test_f1_op"] is not None],
        key=lambda row: row["train_set_usage"],
    )

    print("Top tuning runs by best validation F1(op)")
    for idx, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{idx:>2}. {row['config']} | status={row['status']} | "
            f"best_val={row['best_val_f1_op']:.4f} @ epoch {row['best_epoch']} | "
            f"test_at_best_val={fmt_metric(row['test_f1_op_at_best_val'])}"
        )

    if ranked:
        best = ranked[0]
        print("\nBest tuning config by validation F1(op):")
        print(best["config"])
        print(f"Validation F1(op): {best['best_val_f1_op']:.4f} at epoch {best['best_epoch']}")
        if best["test_f1_op_at_best_val"] is not None:
            print(f"Corresponding test F1(op): {best['test_f1_op_at_best_val']:.4f}")
        if best["resolved_config_path"]:
            print(f"Recommended config for retraining YAML: {best['resolved_config_path']}")
        if best["best_checkpoint_path"]:
            print(f"Recommended config for retraining checkpoint: {best['best_checkpoint_path']}")

    if utilization_rows:
        print("\nTrain-set utilization results")
        for row in utilization_rows:
            print(
                f"usage={row['train_set_usage']:.1f} | "
                f"config={row['config']} | last_test_f1_op={row['last_test_f1_op']:.4f}"
            )

    if ranked and not utilization_rows:
        best = ranked[0]
        print("\nRecommended config for retraining:")
        print(
            f"{best['config']} | checkpoint={best['best_checkpoint_path']}"
        )

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
