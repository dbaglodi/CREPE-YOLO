import os
import glob
import re
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

# Internal project imports
from dataset import MusicNoteDataset
from model import DualStreamMusicYOLO
from loss import MusicYOLOLoss
from utils import music_yolo_collate_fn, get_train_test_split

def get_latest_checkpoint(checkpoint_dir):
    """Finds the checkpoint file with the highest epoch number in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "crepe_yolo_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers using regex: 'crepe_yolo_epoch_50.pt' -> 50
    epochs = []
    for f in checkpoint_files:
        match = re.search(r'epoch_(\d+).pt', f)
        if match:
            epochs.append((int(match.group(1)), f))
    
    if not epochs:
        return None
        
    # Return the path of the one with the maximum epoch number
    return max(epochs, key=lambda x: x[0])[1]

def train(resume=False):
    # --- 1. Hardware Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Training Session Started ---")
    print(f"Target Device: {device}")

    # --- 2. Configuration ---
    checkpoint_dir = 'checkpoints'
    processed_dir = 'processed/itm_flute' 
    batch_size = 4
    num_epochs = 150     # Target goal
    learning_rate = 1e-4
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 3. Data Preparation ---
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Data directory {processed_dir} not found. Run precompute_features.py first.")
        
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    train_stems, _ = get_train_test_split(all_stems, test_size=0.2, seed=42)
    
    dataset = MusicNoteDataset(processed_dir=processed_dir, stems=train_stems)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=music_yolo_collate_fn,
        num_workers=0 # Stable for both Mac/Linux
    )

    # --- 4. Model & Optimizer Initialization ---
    model = DualStreamMusicYOLO(num_anchors=3).to(device)
    loss_fn = MusicYOLOLoss(num_anchors=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # --- 5. Resume Logic (Weights + MLflow Run ID) ---
    start_epoch = 0
    active_run_id = None
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)

    if latest_ckpt and resume:
        print(f"📦 Resuming from checkpoint: {latest_ckpt}")
        # map_location ensures we can move between Mac (MPS) and Linux (CUDA)
        checkpoint = torch.load(latest_ckpt, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        active_run_id = checkpoint.get('run_id') # Retrieve the original MLflow run ID
        print(f"✅ Successfully loaded. Restarting at Epoch {start_epoch}")
    else:
        reason = "Resume not requested" if not resume else "No previous checkpoints found"
        print(f"🌱 Starting fresh training run. ({reason})")

    # --- 6. MLflow Tracking ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CREPE-YOLO-Transcription")
    
    # If active_run_id is None, a new run starts. If it exists, it re-opens the old one.
    with mlflow.start_run(run_id=active_run_id) as run:
        current_run_id = run.info.run_id
        if active_run_id:
            print(f"📈 Continuing MLflow Run: {current_run_id}")
        else:
            print(f"🚀 New MLflow Run ID: {current_run_id}")
            mlflow.log_params({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "device": str(device)
            })

        # --- 7. Main Training Loop ---
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_obj_loss = 0.0
            epoch_box_loss = 0.0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                targets = batch['targets'].to(device)
                features = {k: v.to(device) for k, v in batch['features'].items()}

                optimizer.zero_grad()

                predictions = model(
                    features['posteriorgram'], 
                    features['embedding'], 
                    features['confidence'], 
                    features['gradient']
                )

                # Loss Calculation
                loss_dict = loss_fn(predictions, targets)

                loss_dict = loss_fn(predictions, targets)
                loss = loss_dict['total_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Metrics Tracking
                epoch_loss += loss.item()
                epoch_obj_loss += loss_dict['loss_obj'].item()
                epoch_box_loss += loss_dict['loss_box'].item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.2f}", 
                    'Obj': f"{loss_dict['loss_obj'].item():.2f}"
                })

            # End of Epoch Stats
            avg_loss = epoch_loss / len(dataloader)
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "loss_objectness": epoch_obj_loss / len(dataloader),
                "loss_box_coord": epoch_box_loss / len(dataloader)
            }, step=epoch + 1)

            # --- 8. Periodic Checkpointing ---
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"crepe_yolo_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'run_id': current_run_id, # CRITICAL: Saves the run ID so charts connect
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                
                mlflow.log_artifact(ckpt_path)
                print(f"💾 Saved checkpoint to {ckpt_path}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Resume CREPE-YOLO")
    parser.add_argument('--resume', action='store_true', help="Try to resume from the latest checkpoint if it exists")
    args = parser.parse_args()

    train(resume=args.resume)