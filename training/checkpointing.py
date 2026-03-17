from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    cfg: Dict[str, Any],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": cfg,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
