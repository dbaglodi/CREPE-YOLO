# loss.py
import torch
import torch.nn as nn

class MusicYOLOLoss(nn.Module):
    def __init__(self, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Define extreme aspect ratio anchors for notes (Width, Height)
        # Normalized to the grid size (e.g., width 0.1 = 10% of the audio length)
        # Height is fixed to roughly 1 semitone (5 bins / 360 bins = 0.0138)
        self.register_buffer('anchors', torch.tensor([
            [0.05, 0.0138], # Short note (e.g., 8th note)
            [0.10, 0.0138], # Medium note (e.g., quarter note)
            [0.30, 0.0138]  # Long note (e.g., whole note/sustained)
        ]))

        # Loss weighting
        self.lambda_coord = 5.0 # Penalize box coordinate errors heavily
        self.lambda_noobj = 0.5 # Less penalty for empty cells to balance classes

    def build_targets(self, targets, B, H, T, device):
        """
        Maps the continuous normalized [0, 1] ground truth targets to the exact 
        grid cells (i, j) and assigns them to the best-fitting anchor.
        """
        # Create empty target tensors
        obj_mask = torch.zeros(B, self.num_anchors, H, T, device=device, dtype=torch.bool)
        noobj_mask = torch.ones(B, self.num_anchors, H, T, device=device, dtype=torch.bool)
        tx = torch.zeros(B, self.num_anchors, H, T, device=device)
        ty = torch.zeros(B, self.num_anchors, H, T, device=device)
        tw = torch.zeros(B, self.num_anchors, H, T, device=device)
        th = torch.zeros(B, self.num_anchors, H, T, device=device)

        for b in range(B):
            batch_targets = targets[b]
            # Filter out the padding (-1.0) we added in the collate_fn
            valid_targets = batch_targets[batch_targets[:, 0] != -1]
            
            for target in valid_targets:
                _, gx, gy, gw, gh = target # Class, x, y, w, h
                
                # Scale coordinates to the feature map grid size (H, T)
                gi = int(gx * T)
                gj = int(gy * H)
                
                # Clamp to prevent out-of-bounds indexing
                gi = max(0, min(gi, T - 1))
                gj = max(0, min(gj, H - 1))
                
                # Find the anchor that best matches the ground truth width
                # (Since height is basically constant for notes, we just match width)
                anchor_ious = torch.abs(self.anchors[:, 0] - gw)
                best_n = torch.argmin(anchor_ious)
                
                # Assign to masks
                obj_mask[b, best_n, gj, gi] = True
                noobj_mask[b, best_n, gj, gi] = False
                
                # Calculate the exact regression targets
                # Offset from the top-left of the grid cell
                tx[b, best_n, gj, gi] = (gx * T) - gi
                ty[b, best_n, gj, gi] = (gy * H) - gj
                
                # Log scale for width and height relative to the anchor
                tw[b, best_n, gj, gi] = torch.log(gw / self.anchors[best_n, 0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / self.anchors[best_n, 1] + 1e-16)

        return obj_mask, noobj_mask, tx, ty, tw, th

    def forward(self, predictions, targets):
        """
        predictions: (B, anchors * 5, H, T)
        targets: (B, max_notes, 5)
        """
        B, C, H, T = predictions.shape
        device = predictions.device
        
        # Reshape predictions to separate anchors and coordinate values
        # Shape: (B, anchors, 5, H, T)
        predictions = predictions.view(B, self.num_anchors, 5, H, T)
        
        # Split into coordinates and objectness scores
        pred_x = torch.sigmoid(predictions[..., 0, :, :]) # Sigmoid forces between 0-1 (offset in cell)
        pred_y = torch.sigmoid(predictions[..., 1, :, :])
        pred_w = predictions[..., 2, :, :]
        pred_h = predictions[..., 3, :, :]
        pred_obj = predictions[..., 4, :, :] # Raw logits for BCE loss
        
        # Build targets
        obj_mask, noobj_mask, tx, ty, tw, th = self.build_targets(targets, B, H, T, device)
        
        # --- 1. Objectness Loss (Is there a note here?) ---
        # BCE with Logits for numerical stability
        loss_obj_real = self.bce_loss(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask]))
        loss_obj_fake = self.bce_loss(pred_obj[noobj_mask], torch.zeros_like(pred_obj[noobj_mask]))
        
        loss_obj = loss_obj_real.sum() + self.lambda_noobj * loss_obj_fake.sum()
        
        # --- 2. Coordinate Loss (How accurate is the bounding box?) ---
        # We only calculate coordinate loss if a note actually exists in that cell (obj_mask)
        loss_x = self.mse_loss(pred_x[obj_mask], tx[obj_mask]).sum()
        loss_y = self.mse_loss(pred_y[obj_mask], ty[obj_mask]).sum()
        loss_w = self.mse_loss(pred_w[obj_mask], tw[obj_mask]).sum()
        loss_h = self.mse_loss(pred_h[obj_mask], th[obj_mask]).sum()
        
        loss_box = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
        
        # Total Loss
        total_loss = loss_obj + loss_box
        
        # Normalize by batch size to keep learning rate stable
        total_loss = total_loss / B
        
        return {
            'total_loss': total_loss,
            'loss_obj': loss_obj / B,
            'loss_box': loss_box / B
        }