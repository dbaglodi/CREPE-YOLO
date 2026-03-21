import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add project root to path for training.* imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training.model import DualStreamMusicYOLO
from training.utils import decode_predictions, load_yaml, get_train_test_split

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add project root to path for training.* imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training.model import DualStreamMusicYOLO
from training.utils import decode_predictions, load_yaml, get_train_test_split

def generate_paper_teaser(stem_name, checkpoint_path, processed_dir='data/processed/itm_flute'):
    print(f"🎨 Generating 2x2 paper teaser for {stem_name}...")
    
    # --- 1. Setup & Load Data ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    stem_dir = Path(processed_dir) / stem_name
    
    features_path = stem_dir / 'features.pt'
    notes_path = stem_dir / 'notes.json'
    
    if not features_path.exists() or not notes_path.exists():
        raise FileNotFoundError(f"Missing features or notes for {stem_name} in {processed_dir}")
        
    features = torch.load(features_path, map_location=device)
    
    posteriorgram = features['posteriorgram'].to(device)
    if posteriorgram.dim() == 3:
        posteriorgram = posteriorgram.unsqueeze(1)
        
    embedding = features['embedding'].to(device)
    confidence = features['confidence'].to(device)
    gradient = features['gradient'].to(device)
    
    with open(notes_path, 'r') as f:
        ground_truth = json.load(f)

    # --- 2. Load Model & Forward Pass ---
    model = DualStreamMusicYOLO(num_anchors=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        predictions = model(posteriorgram, embedding, confidence, gradient)
        
    # --- 3. Extract the Objectness Map and Bounding Boxes Safely ---
    B, C, grid_y, grid_t = predictions.shape
    predictions_reshaped = predictions.view(B, 3, 5, grid_y, grid_t)
    
    # Objectness activation map
    obj_map = torch.sigmoid(predictions_reshaped[0, :, 4, :, :])
    obj_heatmap = torch.max(obj_map, dim=0)[0].cpu().numpy()

    # Create boxes using utils.decode_predictions
    anchors = torch.tensor([[0.0026, 0.0139],[0.0062, 0.0139],[0.0153, 0.0139]], device=device)
    decoded_boxes_batch = decode_predictions(predictions, anchors, conf_threshold=0.4, nms_iou_threshold=0.4)
    predictions_decoded = decoded_boxes_batch[0]

    # --- 4. Plotting Setup (2x2 Grid - Slim Profile) ---
    post_gram = posteriorgram.squeeze().cpu().numpy()
    max_time_sec = post_gram.shape[1] * 0.01 
    
   # Reduced figsize width from 20 to 18
    fig, axes = plt.subplots(2, 2, figsize=(18, 7), sharex=True)

    # Lowered wspace from 0.5 to 0.3 for less middle white space
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    time_extent = [0, max_time_sec]
    pitch_extent = [0, 360]

    # [0,0] Top Left: Input
    axes[0,0].set_title("Input Representation (CREPE Posteriorgram)", fontsize=11, fontweight='bold')
    im1 = axes[0,0].imshow(post_gram, aspect='auto', origin='lower', cmap='magma', extent=time_extent + pitch_extent)
    axes[0,0].set_ylabel("Pitch Bins")
    
    # [0,1] Top Right: Ground Truth
    axes[0,1].set_title("Ground Truth Note Bounding Boxes", fontsize=11, fontweight='bold')
    axes[0,1].imshow(post_gram, aspect='auto', origin='lower', cmap='gray', extent=time_extent + pitch_extent)
    axes[0,1].set_ylabel("Pitch Bins")
    
    for note in ground_truth:
        width_sec = note['offset'] - note['onset']
        pitch_y = (note['pitch_midi'] - 24) * 5 
        height_bins = 10 
        rect = patches.Rectangle((note['onset'], pitch_y - height_bins/2), width_sec, height_bins, linewidth=1, edgecolor='limegreen', facecolor='none', alpha=0.9)
        axes[0,1].add_patch(rect)

    # [1,0] Bottom Left: Model Objectness
    axes[1,0].set_title("Dual-Stream YOLO Internal Objectness Activation", fontsize=11, fontweight='bold')
    im3 = axes[1,0].imshow(obj_heatmap, aspect='auto', origin='lower', cmap='viridis', extent=time_extent + [0, 11])
    axes[1,0].set_ylabel("Grid Y") 
    axes[1,0].set_xlabel("Time (Seconds)", fontsize=10)

    # [1,1] Bottom Right: Predictions
    axes[1,1].set_title("Interpreted Model Predictions (Final Bounding Boxes)", fontsize=11, fontweight='bold')
    axes[1,1].imshow(post_gram, aspect='auto', origin='lower', cmap='gray', extent=time_extent + pitch_extent)
    axes[1,1].set_ylabel("Pitch Bins")
    axes[1,1].set_xlabel("Time (Seconds)", fontsize=10)

    for box in predictions_decoded:
        box = box.cpu().numpy()
        cx, cy, w_norm, h_norm, conf, cls = box
        width_sec = w_norm * max_time_sec
        onset_sec = (cx - w_norm / 2.0) * max_time_sec
        height_bins = h_norm * 360.0
        bottom_y_bin = (cy * 360.0) - (height_bins / 2.0)
        
        rect = patches.Rectangle((onset_sec, bottom_y_bin), width_sec, height_bins, linewidth=1, edgecolor='cyan', facecolor='none', alpha=0.9)
        axes[1,1].add_patch(rect)
    
    # Adjusted Colorbars for the slim profile
    cb1 = fig.colorbar(im1, ax=axes[0,0], fraction=0.03, pad=0.04)
    cb1.set_label('Pitch Prob', rotation=270, labelpad=12, fontsize=8)
    
    cb3 = fig.colorbar(im3, ax=axes[1,0], fraction=0.03, pad=0.04)
    cb3.set_label('Objectness', rotation=270, labelpad=12, fontsize=8)
    
    out_file = f"saved_visuals/paper_teaser_slim_grid_{stem_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved slim 2x2 grid teaser to {out_file}")
    plt.show()

def get_sample_test_stem(processed_dir='data/processed/itm_flute', seed=42):
    """Deterministically identifies a completely unseen test-set stem."""
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")
    all_stems = [d.name for d in processed_path.iterdir() if d.is_dir()]
    # Re-use exact train/test splitting logic
    _, test_stems = get_train_test_split(all_stems, test_size=0.2, seed=seed)
    # Using Python's system random ensures a new valid stem each run
    import random
    return random.choice(test_stems)

if __name__ == "__main__":
    ckpt_file = "/Users/skessler/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Semesters/Spring 2026/sandbox/mlruns/215425903410791788/b755f00b9ee142439d0555ae963ddc19/artifacts/crepe_yolo_epoch_20.pt" 
    data_dir = "data/processed/itm_flute"
    
    # Automatically select a confirmed test set stem for evaluation
    target_stem = get_sample_test_stem(processed_dir=data_dir)
    
    generate_paper_teaser(
        stem_name=target_stem, 
        checkpoint_path=ckpt_file,
        processed_dir=data_dir 
    )