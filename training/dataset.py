import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Add training directory to path for local imports
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

class MusicNoteDataset(Dataset):
    """PyTorch Dataset loading pre-computed CREPE features and YOLO targets."""
    def __init__(self, processed_dir: str, stems: list[str]):
        """
        Args:
            processed_dir: Path to the processed data directory.
            stems: List of audio stems to include in this dataset split.
        """
        self.processed_dir = processed_dir
        self.stems = stems
        self.step_size_ms = 10.0 # Standard CREPE step size used during extraction

    def __len__(self) -> int:
        return len(self.stems)

    def _notes_to_yolo_targets(self, notes: list[dict], total_time_sec: float) -> torch.Tensor:
        """
        Converts ground truth to normalized YOLO format: [class_id, x_c, y_c, w, h].
        """
        targets = []
        for n in notes:
            onset = n['onset']
            offset = n['offset']
            pitch_midi = n['pitch_midi']
            
            # 1. Clamp pitch to CREPE's representable range (MIDI 24 to 95)
            pitch_midi = max(24, min(95, pitch_midi))
            
            # 2. X-axis (Time) Normalization
            x_center = ((onset + offset) / 2.0) / total_time_sec
            width = (offset - onset) / total_time_sec
            
            # 3. Y-axis (Pitch) Normalization
            # Center of the note is at bin index: (pitch_midi - 24) * 5 + 2.5
            y_center_bin = (pitch_midi - 24) * 5.0 + 2.5
            y_center = y_center_bin / 360.0
            
            # Height: 1 semitone tall (5 bins)
            height = 5.0 / 360.0
            
            # Class ID is 0 (Monophonic Note)
            targets.append([0.0, x_center, y_center, width, height])
            
        if not targets:
            return torch.zeros((0, 5))
            
        return torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        stem_dir = os.path.join(self.processed_dir, stem)
        
        # 1. Load Pre-computed Features
        features_path = os.path.join(stem_dir, 'features.pt')
        features = torch.load(features_path, weights_only=True) 
        
        # 2. Load Ground Truth Notes
        notes_path = os.path.join(stem_dir, 'notes.json')
        with open(notes_path, 'r') as f:
            notes = json.load(f)
            
        # 3. Calculate Duration and Targets
        T_padded = features['posteriorgram'].shape[-1]
        total_time_sec = T_padded * (self.step_size_ms / 1000.0)
        
        targets = self._notes_to_yolo_targets(notes, total_time_sec)
        
        return {
            'stem': stem,
            'features': features,
            'targets': targets,
            # --- Added for Evaluation ---
            'notes': notes,
            'total_time_sec': total_time_sec
        }