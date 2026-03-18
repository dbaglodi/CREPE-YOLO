# utils.py
import torch
import torch.nn.functional as F
import torchvision
import random
import yaml
import numpy as np

def get_train_test_split(stems, test_size=0.2, seed=42):
    """Deterministically splits stems into train and test sets."""
    sorted_stems = sorted(stems) 
    rng = random.Random(seed)
    rng.shuffle(sorted_stems)
    
    split_idx = int(len(sorted_stems) * (1 - test_size))
    train_stems = sorted_stems[:split_idx]
    test_stems = sorted_stems[split_idx:]
    
    return train_stems, test_stems

def music_yolo_collate_fn(batch):
    """Pads variable-length features and targets for batching."""
    stems = [item['stem'] for item in batch]
    
    # 1. Find the maximum Time dimension (T) in this specific batch
    max_T = max(item['features']['posteriorgram'].shape[-1] for item in batch)
    
    features = {k: [] for k in batch[0]['features'].keys()}
    targets_list = []
    
    # 2. Pad features and rescale targets for each item
    for item in batch:
        T_current = item['features']['posteriorgram'].shape[-1]
        pad_len = max_T - T_current
        
        # Pad features with zeros on the right
        for k in features.keys():
            v = item['features'][k]
            if pad_len > 0:
                v = F.pad(v, (0, pad_len))
            features[k].append(v)
            
        # Adjust YOLO target coordinates based on the new padded length
        targets = item['targets'].clone()
        if pad_len > 0 and len(targets) > 0:
            # Scale x_center and width to maintain absolute time position
            targets[:, 1] *= (T_current / max_T) 
            targets[:, 3] *= (T_current / max_T) 
        targets_list.append(targets)
        
    # 3. Concatenate the padded features into a single batch tensor
    for k in features.keys():
        features[k] = torch.cat(features[k], dim=0)
        # Ensure posteriorgram has the Channel dimension (B, 1, H, T)
        if k == 'posteriorgram' and features[k].dim() == 3:
            features[k] = features[k].unsqueeze(1)
            
    # 4. Pad the targets list to the max number of notes in this batch (-1.0 mask)
    max_notes = max(t.shape[0] for t in targets_list) if targets_list else 0
    padded_targets = []
    for t in targets_list:
        pad_size = max_notes - t.shape[0]
        if pad_size > 0:
            pad = torch.full((pad_size, 5), -1.0, dtype=torch.float32)
            t = torch.cat([t, pad], dim=0)
        padded_targets.append(t)
        
    if padded_targets:
        padded_targets = torch.stack(padded_targets, dim=0)
    else:
        padded_targets = torch.zeros(0)
        
    return {'stems': stems, 'features': features, 'targets': padded_targets}

def decode_predictions(predictions, anchors, conf_threshold=0.5, nms_iou_threshold=0.4):
    """Decodes raw YOLO predictions, applies confidence thresholding, and runs NMS."""
    B, C, H, T = predictions.shape
    num_anchors = len(anchors)
    device = predictions.device
    
    preds = predictions.view(B, num_anchors, 5, H, T)
    
    pred_x = torch.sigmoid(preds[..., 0, :, :])
    pred_y = torch.sigmoid(preds[..., 1, :, :])
    pred_w = preds[..., 2, :, :]
    pred_h = preds[..., 3, :, :]
    pred_conf = torch.sigmoid(preds[..., 4, :, :]) 
    
    batch_boxes = []
    
    for b in range(B):
        mask = pred_conf[b] > conf_threshold
        if not mask.any():
            batch_boxes.append(torch.zeros((0, 6), device=device))
            continue
            
        anchor_idx, grid_y, grid_x = torch.where(mask)
        
        x_offset = pred_x[b, anchor_idx, grid_y, grid_x]
        y_offset = pred_y[b, anchor_idx, grid_y, grid_x]
        w_log = pred_w[b, anchor_idx, grid_y, grid_x]
        h_log = pred_h[b, anchor_idx, grid_y, grid_x]
        scores = pred_conf[b, anchor_idx, grid_y, grid_x]
        
        cx = (grid_x + x_offset) / T
        cy = (grid_y + y_offset) / H
        
        a_w = anchors[anchor_idx, 0].to(device)
        a_h = anchors[anchor_idx, 1].to(device)
        w = torch.exp(w_log) * a_w
        h = torch.exp(h_log) * a_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        keep_idx = torchvision.ops.nms(
            boxes.cpu(), 
            scores.cpu(), 
            nms_iou_threshold
        ).to(device)
        
        surviving_boxes = torch.stack([cx[keep_idx], cy[keep_idx], w[keep_idx], h[keep_idx], scores[keep_idx], torch.zeros_like(scores[keep_idx])], dim=1)
        batch_boxes.append(surviving_boxes)
        
    return batch_boxes

def boxes_to_midi_notes(boxes: torch.Tensor, total_time_sec: float) -> list:
    """Converts decoded YOLO boxes back to MIR-eval friendly note dictionaries."""
    notes = []
    for box in boxes:
        cx, cy, w, h, conf, cls = box.cpu().tolist()
        
        onset = (cx - w / 2.0) * total_time_sec
        offset = (cx + w / 2.0) * total_time_sec
        
        y_center_bin = cy * 360.0
        pitch_midi = ((y_center_bin - 2.5) / 5.0) + 24
        
        if offset > onset:
            notes.append({
                'onset': round(max(0, onset), 4),
                'offset': round(offset, 4),
                'pitch_midi': int(round(pitch_midi)),
                'confidence': round(conf, 4)
            })
            
    return sorted(notes, key=lambda x: x['onset'])


def load_yaml(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(config_path: str, config: dict) -> None:
    """Save a YAML configuration file."""
    import os
    from pathlib import Path
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)