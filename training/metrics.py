from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


def decode_predictions(outputs: torch.Tensor, cfg: dict) -> List[dict]:
    """
    Placeholder decoder.

    Later, replace with note-event decoding:
    - confidence thresholding
    - onset/offset extraction
    - pitch decoding
    - note merge/suppression
    """
    preds = []
    for row in outputs.detach().cpu():
        preds.append(
            {
                "max_index": int(torch.argmax(row).item()),
                "mean_score": float(torch.mean(row).item()),
            }
        )
    return preds


@dataclass
class RunningAverageMeter:
    values: List[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self.values.append(float(value))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))


def batch_dummy_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Placeholder metric: negative MSE transformed into a 'higher is better' score.
    """
    mse = torch.mean((outputs - targets) ** 2).item()
    return float(1.0 / (1.0 + mse))


def summarize_validation(losses: List[float], scores: List[float]) -> Dict[str, float]:
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "val_loss": float(avg_loss),
        "val_score": float(avg_score),
    }
