# evaluate.py
import os
import json
import torch
import numpy as np
import mir_eval
from tqdm import tqdm

from dataset import MusicNoteDataset
from model import DualStreamMusicYOLO
from utils import decode_predictions, boxes_to_midi_notes, get_train_test_split

def notes_to_mir_arrays(notes):
    """Converts a list of note dicts to NumPy arrays for mir_eval."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n['onset'], n['offset']] for n in notes], dtype=float)
    p  = np.array([float(n['pitch_midi']) for n in notes], dtype=float)
    return iv, p

def evaluate():
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Starting Evaluation on: {device} ---")
    
    # Update this to point to your best checkpoint once trained
    checkpoint_path = 'checkpoints/crepe_yolo_epoch_50.pt'
    processed_dir = 'processed/itm_flute'
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint {checkpoint_path} not found. Train the model first!")
        return

    # mir_eval tolerances (Matching CREPE Notes / MT3 papers)
    ONSET_TOL    = 0.05   # 50 ms
    OFFSET_RATIO = 0.20   # 20% of note duration
    OFFSET_MIN   = 0.05   # 50 ms minimum
    PITCH_TOL    = 0.5    # +/- 0.5 semitones
    
    # --- 2. Load Model ---
    model = DualStreamMusicYOLO(num_anchors=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    anchors = torch.tensor([
        [0.05, 0.0138],
        [0.10, 0.0138],
        [0.30, 0.0138]
    ], device=device)

    # --- 3. Initialize Dataset ---
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    # Grab the exact same split, but only use the test set
    _, test_stems = get_train_test_split(all_stems, test_size=0.2, seed=42)
    print(f"Evaluating on {len(test_stems)} unseen test samples.")
    
    dataset = MusicNoteDataset(processed_dir=processed_dir, stems=test_stems)
    
    results = []
    
    # --- 4. Evaluation Loop ---
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            item = dataset[idx]
            stem = item['stem']
            
            # Prepare inputs
            # They already have the batch dimension (dim=0) from precompute_features.py
            features = {k: v.to(device) for k, v in item['features'].items()}
            
            # Ensure posteriorgram has the Channel dimension (B, 1, H, T) for the 2D CNN
            if features['posteriorgram'].dim() == 3:
                features['posteriorgram'] = features['posteriorgram'].unsqueeze(1)
            
            # Forward Pass
            predictions = model(
                features['posteriorgram'], 
                features['embedding'], 
                features['confidence'], 
                features['gradient']
            )
            
            # Decode Boxes
            batch_boxes = decode_predictions(predictions, anchors, conf_threshold=0.5, nms_iou_threshold=0.4)
            
            # Calculate total time
            T_padded = features['posteriorgram'].shape[-1]
            total_time_sec = T_padded * (dataset.step_size_ms / 1000.0)
            
            # Convert to MIDI dicts
            est_notes = boxes_to_midi_notes(batch_boxes[0], total_time_sec)
            
            # Load Ground Truth
            notes_path = os.path.join(processed_dir, stem, 'notes.json')
            with open(notes_path, 'r') as f:
                ref_notes = json.load(f)
                
            # mir_eval Metrics
            ref_iv, ref_p = notes_to_mir_arrays(ref_notes)
            est_iv, est_p = notes_to_mir_arrays(est_notes)
            
            if len(ref_iv) == 0 and len(est_iv) == 0:
                p, r, f, aor = 1.0, 1.0, 1.0, 1.0
                p_op, r_op, f_op = 1.0, 1.0, 1.0
            else:
                # Strict: onset + offset + pitch
                p, r, f, aor = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_iv, ref_p, est_iv, est_p,
                    onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL,
                    offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN,
                )
                
                # Lenient: onset + pitch only (Ignore offset bounds)
                p_op, r_op, f_op, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_iv, ref_p, est_iv, est_p,
                    onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL,
                    offset_ratio=None, offset_min_tolerance=None
                )
                
            results.append({
                'stem': stem,
                'P': p, 'R': r, 'F1': f, 'AOR': aor,
                'P_op': p_op, 'R_op': r_op, 'F1_op': f_op,
                'n_ref': len(ref_notes), 'n_est': len(est_notes)
            })

    # --- 5. Aggregate and Print ---
    avg_p = np.mean([r['P'] for r in results])
    avg_r = np.mean([r['R'] for r in results])
    avg_f = np.mean([r['F1'] for r in results])
    avg_aor = np.mean([r['AOR'] for r in results])
    
    avg_p_op = np.mean([r['P_op'] for r in results])
    avg_r_op = np.mean([r['R_op'] for r in results])
    avg_f_op = np.mean([r['F1_op'] for r in results])
    
    # Print mimicking the baseline notebook format
    print("\n")
    hdr = f"  {'Model':<16} {'P':>6} {'R':>6} {'F1':>6} {'AOR':>6}   {'P(op)':>6} {'R(op)':>6} {'F1(op)':>7}"
    print('=' * len(hdr))
    print(hdr)
    print(f"  {'':16} {'':6} {'':6} {'':6} {'':6}   {'<-- onset+pitch only':>21}")
    print('=' * len(hdr))
    
    print(f"  {'CREPE-YOLO':<16} {avg_p:>6.3f} {avg_r:>6.3f} {avg_f:>6.3f} "
          f"{avg_aor:>6.3f}   {avg_p_op:>6.3f} {avg_r_op:>6.3f} {avg_f_op:>7.3f}")
          
    print('=' * len(hdr))
    print(f'  Left: onset+offset+pitch  |  Tolerances: onset +/-{ONSET_TOL*1000:.0f}ms, '
          f'offset +/-{OFFSET_MIN*1000:.0f}ms or {OFFSET_RATIO*100:.0f}%, pitch +/-{PITCH_TOL} st')

if __name__ == "__main__":
    evaluate()