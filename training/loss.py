# loss.py
import torch
import torch.nn as nn


class MusicYOLOXLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def build_targets(self, targets, batch_size, height, time_steps, device):
        """
        Assign each note to the cell containing its center and keep at most one
        positive per cell. When multiple notes land on the same cell, retain the
        larger note box as the target for that location.
        """
        obj_mask = torch.zeros(batch_size, height, time_steps, device=device, dtype=torch.bool)
        noobj_mask = torch.ones(batch_size, height, time_steps, device=device, dtype=torch.bool)
        tx = torch.zeros(batch_size, height, time_steps, device=device)
        ty = torch.zeros(batch_size, height, time_steps, device=device)
        tw = torch.zeros(batch_size, height, time_steps, device=device)
        th = torch.zeros(batch_size, height, time_steps, device=device)
        best_area = torch.full((batch_size, height, time_steps), -1.0, device=device)

        for batch_idx in range(batch_size):
            batch_targets = targets[batch_idx]
            valid_targets = batch_targets[batch_targets[:, 0] != -1]

            for target in valid_targets:
                _, gx, gy, gw, gh = target

                grid_x = int(gx * time_steps)
                grid_y = int(gy * height)
                grid_x = max(0, min(grid_x, time_steps - 1))
                grid_y = max(0, min(grid_y, height - 1))

                area = gw * gh
                if area <= best_area[batch_idx, grid_y, grid_x]:
                    continue

                best_area[batch_idx, grid_y, grid_x] = area
                obj_mask[batch_idx, grid_y, grid_x] = True
                noobj_mask[batch_idx, grid_y, grid_x] = False
                tx[batch_idx, grid_y, grid_x] = (gx * time_steps) - grid_x
                ty[batch_idx, grid_y, grid_x] = (gy * height) - grid_y
                tw[batch_idx, grid_y, grid_x] = gw
                th[batch_idx, grid_y, grid_x] = gh

        return obj_mask, noobj_mask, tx, ty, tw, th

    def forward(self, predictions, targets):
        batch_size, channels, height, time_steps = predictions.shape
        if channels != 5:
            raise ValueError(
                f"YOLOX predictions must have 5 channels, received {channels}"
            )

        device = predictions.device
        pred_x = torch.sigmoid(predictions[:, 0, :, :])
        pred_y = torch.sigmoid(predictions[:, 1, :, :])
        pred_w = torch.sigmoid(predictions[:, 2, :, :])
        pred_h = torch.sigmoid(predictions[:, 3, :, :])
        pred_obj = predictions[:, 4, :, :]

        obj_mask, noobj_mask, tx, ty, tw, th = self.build_targets(
            targets,
            batch_size,
            height,
            time_steps,
            device,
        )

        loss_obj_real = self.bce_loss(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask]))
        loss_obj_fake = self.bce_loss(
            pred_obj[noobj_mask],
            torch.zeros_like(pred_obj[noobj_mask]),
        )
        loss_obj = loss_obj_real.sum() + self.lambda_noobj * loss_obj_fake.sum()

        loss_x = self.mse_loss(pred_x[obj_mask], tx[obj_mask]).sum()
        loss_y = self.mse_loss(pred_y[obj_mask], ty[obj_mask]).sum()
        loss_w = self.mse_loss(pred_w[obj_mask], tw[obj_mask]).sum()
        loss_h = self.mse_loss(pred_h[obj_mask], th[obj_mask]).sum()
        loss_box = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        total_loss = (loss_obj + loss_box) / batch_size
        return {
            "total_loss": total_loss,
            "loss_obj": loss_obj / batch_size,
            "loss_box": loss_box / batch_size,
        }

    def get_decode_config(self, device):
        _ = device
        return {}


def build_loss(yolox_cfg=None):
    yolox_cfg = yolox_cfg or {}
    return MusicYOLOXLoss(
        lambda_coord=yolox_cfg.get("lambda_coord", 5.0),
        lambda_noobj=yolox_cfg.get("lambda_noobj", 0.5),
    )
