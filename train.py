# train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

# Force MLflow to use a local directory instead of a hidden database
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CREPE-YOLO-Transcription")

from dataset import MusicNoteDataset
from model import DualStreamMusicYOLO
from loss import MusicYOLOLoss
from utils import music_yolo_collate_fn, get_train_test_split

def train():
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Starting Training on device: {device} ---")

    # --- 2. Configuration ---
    processed_dir = 'processed/itm_flute' 
    batch_size = 4
    num_epochs = 300
    learning_rate = 1e-4

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    train_stems, test_stems = get_train_test_split(all_stems, test_size=0.2, seed=42)
    print(f"Dataset Split: {len(train_stems)} Training | {len(test_stems)} Validation")

    # --- 3. Initialize Dataset and DataLoader ---
    dataset = MusicNoteDataset(processed_dir=processed_dir, stems=train_stems)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=music_yolo_collate_fn,
        num_workers=0
    )

    # --- 4. Initialize Model, Loss, and Optimizer ---
    model = DualStreamMusicYOLO(num_anchors=3).to(device)
    loss_fn = MusicYOLOLoss(num_anchors=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    os.makedirs('checkpoints', exist_ok=True)

    # =========================================================================
    # MLFLOW INTEGRATION
    # =========================================================================
    mlflow.set_experiment("CREPE-YOLO-Transcription")
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log all hyperparameters for this specific run
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "optimizer": "AdamW",
            "train_samples": len(train_stems),
            "val_samples": len(test_stems)
        })

        # --- 5. Training Loop ---
        for epoch in range(num_epochs):
            model.train()
            
            # Tracking metrics for the epoch average
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

                loss_dict = loss_fn(predictions, targets)
                loss = loss_dict['total_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Accumulate
                epoch_loss += loss.item()
                epoch_obj_loss += loss_dict['loss_obj'].item()
                epoch_box_loss += loss_dict['loss_box'].item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'Obj': f"{loss_dict['loss_obj'].item():.4f}", 
                    'Box': f"{loss_dict['loss_box'].item():.4f}"
                })

            # Calculate and log the epoch averages to MLflow
            num_batches = len(dataloader)
            avg_loss = epoch_loss / num_batches
            avg_obj = epoch_obj_loss / num_batches
            avg_box = epoch_box_loss / num_batches
            
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "loss_objectness": avg_obj,
                "loss_box_coord": avg_box
            }, step=epoch + 1)

            print(f"End of Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

            # Save Checkpoint every 10 epochs and log as an MLflow Artifact
            if (epoch + 1) % 10 == 0:
                ckpt_path = f"checkpoints/crepe_yolo_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, ckpt_path)
                
                # Push the physical model weights to the MLflow dashboard
                mlflow.log_artifact(ckpt_path, artifact_path="model_checkpoints")
                print(f"Saved and logged checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()