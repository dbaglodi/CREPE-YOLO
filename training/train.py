from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.datasets import ManifestAudioDataset
from models.registry import MODEL_REGISTRY
from training.checkpointing import save_checkpoint
from training.losses import compute_loss
from training.metrics import batch_dummy_score, summarize_validation
from training.utils import append_csv_row, ensure_dir, save_json


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    target_dim = cfg["loss"]["target_dim"]

    train_ds = ManifestAudioDataset(
        manifest_path=data_cfg["train_manifest"],
        data_root=data_cfg["data_root"],
        input_num_samples=data_cfg["input_num_samples"],
        target_dim=target_dim,
    )
    val_ds = ManifestAudioDataset(
        manifest_path=data_cfg["val_manifest"],
        data_root=data_cfg["data_root"],
        input_num_samples=data_cfg["input_num_samples"],
        target_dim=target_dim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=bool(data_cfg["pin_memory"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=bool(data_cfg["pin_memory"]),
    )
    return train_loader, val_loader


def build_model(cfg: dict) -> torch.nn.Module:
    model_name = cfg["model"]["name"]
    kwargs = cfg["model"].get("kwargs", {})
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_REGISTRY[model_name](**kwargs)
    return model


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["optim"]
    name = opt_cfg["name"].lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    raise ValueError(f"Unsupported optimizer: {name}")


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    model.eval()
    losses = []
    scores = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["inputs"].to(device)
            y = batch["targets"].to(device)

            outputs = model(x)
            loss, _ = compute_loss(outputs, y, cfg)
            score = batch_dummy_score(outputs, y)

            losses.append(float(loss.detach().cpu().item()))
            scores.append(float(score))

    return summarize_validation(losses, scores)


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["run_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    device_str = cfg["train"]["device"]
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, cfg)

    history_csv = run_dir / "history.csv"
    best_metric = float("-inf")
    best_epoch = -1

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["inputs"].to(device)
            y = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss, loss_dict = compute_loss(outputs, y, cfg)
            loss.backward()

            grad_clip = cfg["train"].get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)

        metrics = {"val_loss": None, "val_score": None}
        if epoch % int(cfg["train"]["validate_every"]) == 0:
            metrics = validate(model, val_loader, device, cfg)
            current_metric = float(metrics["val_score"])

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    cfg=cfg,
                )

        if epoch % int(cfg["train"]["save_every"]) == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                cfg=cfg,
            )

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_score": metrics["val_score"],
            "best_val_score_so_far": best_metric if best_metric != float("-inf") else None,
        }
        append_csv_row(history_csv, log_row)
        print(log_row)

    summary = {
        "run_name": cfg["run_name"],
        "best_epoch": best_epoch,
        "best_val_score": None if best_metric == float("-inf") else best_metric,
        "output_dir": str(run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    return summary
