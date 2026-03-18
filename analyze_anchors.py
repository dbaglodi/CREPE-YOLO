# analyze_anchors.py
import os
import json
import torch
import numpy as np
from sklearn.cluster import KMeans

def analyze_dataset_anchors():
    processed_dir = 'processed/itm_flute'
    step_size_ms = 10.0 # Standard CREPE step size
    
    durations_sec = []
    normalized_widths = []
    
    stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    print(f"Scanning {len(stems)} stems for ground truth notes...")
    
    for stem in stems:
        stem_dir = os.path.join(processed_dir, stem)
        notes_path = os.path.join(stem_dir, 'notes.json')
        features_path = os.path.join(stem_dir, 'features.pt')
        
        if not os.path.exists(notes_path) or not os.path.exists(features_path):
            continue
            
        # 1. We need the total time to normalize the widths correctly
        # This matches the exact math in your dataset.py
        features = torch.load(features_path, map_location='cpu', weights_only=True)
        T_padded = features['posteriorgram'].shape[-1]
        total_time_sec = T_padded * (step_size_ms / 1000.0)
        
        with open(notes_path, 'r') as f:
            notes = json.load(f)
            
        for n in notes:
            # Absolute duration in seconds
            duration = n['offset'] - n['onset']
            durations_sec.append(duration)
            
            # YOLO Normalized Width (0.0 to 1.0)
            w = duration / total_time_sec
            normalized_widths.append(w)

    if not normalized_widths:
        print("No notes found. Check your processed_dir path!")
        return

    durations_sec = np.array(durations_sec)
    
    # K-Means expects a 2D array, so we reshape
    X_widths = np.array(normalized_widths).reshape(-1, 1) 
    
    # In this specific architecture, height is always 1 semitone
    const_height = 5.0 / 360.0 

    print("\n--- 📊 Note Duration Statistics ---")
    print(f"Total notes analyzed: {len(durations_sec):,}")
    print(f"Shortest note:        {np.min(durations_sec)*1000:.1f} ms")
    print(f"Longest note:         {np.max(durations_sec)*1000:.1f} ms")
    print(f"Average note:         {np.mean(durations_sec)*1000:.1f} ms")

    print("\n--- 🎯 Running K-Means Clustering (k=3) ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_widths)
    
    # Extract centers and sort from smallest to largest
    centers = kmeans.cluster_centers_.flatten()
    centers.sort() 

    print("\n✅ OPTIMAL ANCHORS GENERATED:")
    print("Replace the anchors in your code with these exact values:\n")
    
    print("anchors = torch.tensor([")
    for w in centers:
        print(f"    [{w:.4f}, {const_height:.4f}],")
    print("], device=device)")

if __name__ == "__main__":
    analyze_dataset_anchors()