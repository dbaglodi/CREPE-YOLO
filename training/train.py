# train.py
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

# Add training directory to path for local imports
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset import MusicNoteDataset
from model import DualStreamMusicYOLO
from loss import MusicYOLOLoss
from utils import music_yolo_collate_fn, get_train_test_split


def run_training(cfg: dict) -> dict:
    """
    Main training function that accepts a config dictionary.
    
    Args:
        cfg: Configuration dict with keys:
            - seed: Random seed
            - run_name: Name for this run
            - output_root: Root directory for outputs
            - data: dict with processed_dir, batch_size, num_workers
            - model: dict with num_anchors
            - optim: dict with lr, weight_decay, name
            - train: dict with epochs, device, grad_clip, save_every
            - mlflow: dict with tracking_uri, experiment_name
    
    Returns:
        Summary dict with training results
    """
    # --- 1. Setup Device ---
    device_str = cfg.get("train", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Training on device: {device} ---")
    print(f"--- Run Name: {cfg.get('run_name', 'unnamed')} ---")

    # --- 2. Extract Configuration ---
    seed = cfg.get("seed", 42)
    run_name = cfg.get("run_name", "default_run")
    output_root = cfg.get("output_root", "outputs")
    
    # Data config
    processed_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 0)
    test_size = cfg["data"].get("test_size", 0.2)
    
    # Model config
    num_anchors = cfg["model"].get("num_anchors", 3)
    
    # Optimizer config
    learning_rate = cfg["optim"]["lr"]
    weight_decay = cfg["optim"].get("weight_decay", 0.0)
    
    # Training config
    num_epochs = cfg["train"]["epochs"]
    grad_clip = cfg["train"].get("grad_clip", 5.0)
    save_every = cfg["train"].get("save_every", 10)
    
    # MLflow config
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
    mlflow_exp = mlflow_cfg.get("experiment_name", "CREPE-YOLO-Transcription")

    # --- 3. Setup Output Directories ---
    run_dir = Path(output_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Setup MLflow ---
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_exp)

    # --- 5. Verify Dataset ---
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not all_stems:
        raise ValueError(f"No dataset stems found in {processed_dir}")
    
    train_stems, test_stems = get_train_test_split(all_stems, test_size=test_size, seed=seed)
    print(f"Dataset Split: {len(train_stems)} Training | {len(test_stems)} Validation")

    # --- 6. Initialize Dataset and DataLoader ---
    dataset = MusicNoteDataset(processed_dir=processed_dir, stems=train_stems)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=music_yolo_collate_fn,
        num_workers=num_workers
    )

    # --- 7. Initialize Model, Loss, and Optimizer ---
    model = DualStreamMusicYOLO(num_anchors=num_anchors).to(device)
    loss_fn = MusicYOLOLoss(num_anchors=num_anchors).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # =========================================================================
    # MLFLOW INTEGRATION
    # =========================================================================
    training_results = {
        "epochs_completed": 0,
        "final_loss": None,
        "best_loss": float('inf'),
        "checkpoint_path": None
    }
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log all hyperparameters for this specific run
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "optimizer": "AdamW",
            "train_samples": len(train_stems),
            "val_samples": len(test_stems),
            "num_anchors": num_anchors
        })

        # --- 8. Training Loop ---
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
            training_results["epochs_completed"] = epoch + 1
            training_results["final_loss"] = avg_loss
            
            # Track best loss
            if avg_loss < training_results["best_loss"]:
                training_results["best_loss"] = avg_loss

            # Save Checkpoint at specified interval
            if (epoch + 1) % save_every == 0:
                ckpt_path = str(checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': cfg
                }, ckpt_path)
                
                # Log checkpoint to MLflow
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                training_results["checkpoint_path"] = ckpt_path
                print(f"Saved and logged checkpoint: {ckpt_path}")
        
        # Log final artifacts
        mlflow.log_artifact(str(run_dir / "resolved_config.yaml"), artifact_path="config")

    print("\n=== Training Complete ===")
    print(f"Total Epochs: {training_results['epochs_completed']}")
    print(f"Final Loss: {training_results['final_loss']:.4f}")
    print(f"Best Loss: {training_results['best_loss']:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return training_results


if __name__ == "__main__":
    # For standalone execution (not recommended; use run_train.py instead)
    raise RuntimeError("Please use scripts/run_train.py to launch training with proper config.")