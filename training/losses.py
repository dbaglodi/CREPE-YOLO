from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Starter loss.

    Replace this with your real CREPE-YOLO detection/transcription loss later.
    """
    loss_name = cfg["loss"]["name"].lower()

    if loss_name == "mse":
        loss = F.mse_loss(outputs, targets)
    elif loss_name == "l1":
        loss = F.l1_loss(outputs, targets)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return loss, {"total_loss": float(loss.detach().cpu().item())}
