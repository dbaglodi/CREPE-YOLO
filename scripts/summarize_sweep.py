from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def parse_log(path: Path) -> dict:
    config = None
    status = "unknown"
    best_f1 = None
    best_epoch = None
    last_f1 = None
    last_epoch = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"Running sweep task \d+ with config: .*/(sweep_[^/]+)\.ya?ml", line)
        if match:
            config = match.group(1)

        match = re.search(r"End of Epoch (\d+) \| .*?Val F1\(op\): ([0-9.]+)", line)
        if match:
            epoch = int(match.group(1))
            f1 = float(match.group(2))
            last_epoch = epoch
            last_f1 = f1
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch

        if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
            status = "oom"
        elif "--- Training Complete ---" in line and status != "oom":
            status = "completed"

    return {
        "log_path": str(path),
        "config": config or path.stem,
        "status": status,
        "best_val_f1_op": best_f1,
        "best_epoch": best_epoch,
        "last_val_f1_op": last_f1,
        "last_epoch": last_epoch,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.logs_glob))
    if not paths:
        raise SystemExit(f"No logs matched: {args.logs_glob}")

    rows = [parse_log(path) for path in paths]
    ranked = sorted(
        [row for row in rows if row["best_val_f1_op"] is not None],
        key=lambda row: row["best_val_f1_op"],
        reverse=True,
    )

    print("Top runs by best_val_F1_op")
    for idx, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{idx:>2}. {row['config']} | status={row['status']} | "
            f"best={row['best_val_f1_op']:.4f} @ epoch {row['best_epoch']} | "
            f"last={row['last_val_f1_op']:.4f} @ epoch {row['last_epoch']}"
        )

    if ranked:
        best = ranked[0]
        print("\nBest config to keep:")
        print(
            f"{best['config']} with best_val_F1_op={best['best_val_f1_op']:.4f} "
            f"at epoch {best['best_epoch']}"
        )

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
