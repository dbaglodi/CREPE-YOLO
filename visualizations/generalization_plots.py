import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the input files and their corresponding titles
files = [
    'train_eval_R-val_R.csv',
    'train_eval_P-val_P.csv',
    'train_eval_F1-val_F1.csv',
    'train_eval_F1_op-val_F1_op.csv',
    'loss_objectness-val_loss_objectness.csv',
    'loss_box_coord-val_loss_box_coord.csv'
]

titles = [
    'Recall (Strict)',
    'Precision (Strict)',
    'F1 Score (Strict)',
    'F1 Score (Onset+Pitch Only)',
    'Objectness Loss',
    'Box Coordinate Loss'
]

# Set up the 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Define the directory
download_dir = '~/Downloads/'

for i, (file_name, title) in enumerate(zip(files, titles)):
    # --- THE FIX: Expand the tilde and join paths properly ---
    full_path = os.path.expanduser(os.path.join(download_dir, file_name))
    
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found. Skipping.")
        continue
        
    df = pd.read_csv(full_path)
    # Get unique metrics and sort them so Train typically comes before Validation
    metric_names = sorted(df['metric'].unique(), 
                          key=lambda x: 1 if x.lower().startswith('val') else 0)
    
    for m in metric_names:
        subset = df[df['metric'] == m].sort_values('step')
        
        label = "Validation" if m.lower().startswith('val') else "Train"
        color = 'tab:blue' if label == "Train" else 'tab:orange'
        
        axes[i].plot(subset['step'], subset['value'], 
                     label=label, color=color, 
                     linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    # Aesthetics for each subplot
    axes[i].set_title(title, fontsize=14, fontweight='bold', pad=10)
    axes[i].set_xlabel('Epoch', fontsize=10)
    axes[i].set_ylabel('Value', fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].legend(frameon=True, shadow=True)

# Main figure title
plt.suptitle('DualStreamMusicYOLO Metrics: Training and Validation Comparison', 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('saved_visuals/generalization_plots.png', dpi=300, bbox_inches='tight')
print("Successfully generated generalization_plots.png")