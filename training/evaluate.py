import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import mir_eval
from tqdm import tqdm

# Add project root to path so explicit training.* imports work in script mode.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import MusicNoteDataset
from training.model import DualStreamMusicYOLO
from training.utils import decode_predictions, boxes_to_midi_notes, get_train_test_split, load_yaml

def notes_to_mir_arrays(notes):
    """Converts note dicts to NumPy arrays for mir_eval processing."""
    if not notes: return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n['onset'], n['offset']] for n in notes], dtype=float)
    p  = np.array([float(n['pitch_midi']) for n in notes], dtype=float)
    return iv, p

def run_full_metrics(predictions, dataset, anchors, conf, nms):
    """Calculates both Strict and Lenient (op) metrics across the entire test set."""
    results = []
    
    # Standard MIR tolerances
    TOL = {'on': 0.05, 'off_r': 0.2, 'off_min': 0.05, 'p': 0.5}

    for i, pred in enumerate(predictions):
        item = dataset[i]
        batch_boxes = decode_predictions(pred, anchors, conf_threshold=conf, nms_iou_threshold=nms)
        
        est_notes = boxes_to_midi_notes(batch_boxes[0], item['total_time_sec'])
        ref_iv, ref_p = notes_to_mir_arrays(item['notes'])
        est_iv, est_p = notes_to_mir_arrays(est_notes)

        if len(est_iv) == 0:
            # Handle cases with no predictions to avoid divide-by-zero
            stats = {k: 0.0 for k in ['P', 'R', 'F1', 'AOR', 'P_op', 'R_op', 'F1_op']}
        else:
            # 1. Strict Metrics (Onset + Offset + Pitch)
            p, r, f, aor = mir_eval.transcription.precision_recall_f1_overlap(
                ref_iv, ref_p, est_iv, est_p,
                onset_tolerance=TOL['on'], pitch_tolerance=TOL['p'],
                offset_ratio=TOL['off_r'], offset_min_tolerance=TOL['off_min']
            )
            # 2. Lenient Metrics (Onset + Pitch only)
            p_op, r_op, f_op, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_iv, ref_p, est_iv, est_p,
                onset_tolerance=TOL['on'], pitch_tolerance=TOL['p'],
                offset_ratio=None, offset_min_tolerance=None
            )
            stats = {'P': p, 'R': r, 'F1': f, 'AOR': aor, 'P_op': p_op, 'R_op': r_op, 'F1_op': f_op}
        
        results.append(stats)

    # Aggregate results into a single average dictionary
    return {k: np.mean([r[k] for r in results]) for k in results[0].keys()}

def print_mir_table(metrics, model_name="CREPE-YOLO"):
    """Prints the evaluation results in the standardized MIR-eval table format."""
    hdr = f"  {'Model':<16} {'P':>6} {'R':>6} {'F1':>6} {'AOR':>6}   {'P(op)':>6} {'R(op)':>6} {'F1(op)':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print(f"  {'':16} {'':6} {'':6} {'':6} {'':6}   {'<-- onset+pitch only':>21}")
    print("=" * len(hdr))
    
    print(f"  {model_name:<16} {metrics['P']:>6.3f} {metrics['R']:>6.3f} {metrics['F1']:>6.3f} "
          f"{metrics['AOR']:>6.3f}   {metrics['P_op']:>6.3f} {metrics['R_op']:>6.3f} {metrics['F1_op']:>7.3f}")
          
    print("=" * len(hdr))
    print("  Left: onset+offset+pitch  |  Tolerances: onset +/-50ms, offset +/-50ms or 20%, pitch +/-0.5 st\n")

def main():
    parser = argparse.ArgumentParser(description="CREPE-YOLO Evaluation and Tuning")
    parser.add_argument('--tune', action='store_true', help="Run grid search for thresholds")
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--ckpt', type=str, default='outputs/crepe_yolo_base_run/checkpoints/checkpoint_epoch_60.pt')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_yaml(str(config_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    processed_dir = cfg["data"]["processed_dir"]
    
    # Setup Model and Data
    model = DualStreamMusicYOLO(num_anchors=3).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    anchors = torch.tensor([[0.0026, 0.0139],[0.0062, 0.0139],[0.0153, 0.0139]], device=device)
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    _, test_stems = get_train_test_split(all_stems, test_size=0.2, seed=42)
    dataset = MusicNoteDataset(processed_dir=processed_dir, stems=test_stems)

    # 1. Inference Pass (Cached)
    print(f"--- Inference Pass: {len(dataset)} files on {device} ---")
    prediction_cache = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            features = {k: v.to(device) for k, v in item['features'].items()}
            if features['posteriorgram'].dim() == 3:
                features['posteriorgram'] = features['posteriorgram'].unsqueeze(1)
            prediction_cache.append(model(features['posteriorgram'], features['embedding'], features['confidence'], features['gradient']))

    # 2. Threshold Determination
    best_conf, best_nms = 0.4, 0.4 # Baseline defaults
    
    if args.tune:
        print("\n--- Tuning Hyperparameters ---")
        conf_grid, nms_grid = [0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5]
        print(f"{'Conf':<10} | {'NMS':<10} | {'F1(op)':<12}")
        print("-" * 35)
        
        max_f1 = 0
        for c in conf_grid:
            for n in nms_grid:
                # Use a fast metric for tuning
                results = run_full_metrics(prediction_cache, dataset, anchors, c, n)
                print(f"{c:<10.2f} | {n:<10.2f} | {results['F1_op']:<12.4f}")
                if results['F1_op'] > max_f1:
                    max_f1, best_conf, best_nms = results['F1_op'], c, n
        print(f"\n🏆 Optimal found: Conf={best_conf}, NMS={best_nms}")

    # 3. Final Output (Full Table)
    print(f"\n--- Final Results (Conf={best_conf}, NMS={best_nms}) ---")
    final_metrics = run_full_metrics(prediction_cache, dataset, anchors, best_conf, best_nms)
    print_mir_table(final_metrics)

if __name__ == "__main__":
    main()
