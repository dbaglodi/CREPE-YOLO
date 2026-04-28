import os
import glob
import re
import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import MusicNoteDataset
from training.model import DualStreamMusicYOLO
from training.loss import MusicYOLOLoss
from training.utils import load_yaml, music_yolo_collate_fn, get_train_val_test_split

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


def build_eval_subset(dataset, max_items):
    if max_items is None or max_items <= 0 or len(dataset) <= max_items:
        return dataset
    return Subset(dataset, range(max_items))


def evaluate_split(model, loss_fn, dataloader, dataset, device, eval_cfg, prefix):
    """Run loss and MIR-style metrics on a dataset split."""
    if len(dataset) == 0:
        return {}

    model.eval()
    total_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            targets = batch['targets'].to(device)
            features = {k: v.to(device) for k, v in batch['features'].items()}
            predictions = model(
                features['posteriorgram'],
                features['embedding'],
                features['confidence'],
                features['gradient'],
            )
            loss_dict = loss_fn(predictions, targets)
            total_loss += loss_dict['total_loss'].item()
            total_obj += loss_dict['loss_obj'].item()
            total_box += loss_dict['loss_box'].item()
            num_batches += 1

    metrics = {
        f"{prefix}_loss": total_loss / num_batches,
        f"{prefix}_loss_objectness": total_obj / num_batches,
        f"{prefix}_loss_box_coord": total_box / num_batches,
    }

    from training.evaluate import run_full_metrics

    anchors = loss_fn.anchors.detach().to(device)
    conf_threshold = eval_cfg.get("conf_threshold", 0.4)
    nms_iou_threshold = eval_cfg.get("nms_iou_threshold", 0.4)
    prediction_cache = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            features = {k: v.to(device) for k, v in item['features'].items()}
            if features['posteriorgram'].dim() == 3:
                features['posteriorgram'] = features['posteriorgram'].unsqueeze(1)
            prediction_cache.append(
                model(
                    features['posteriorgram'],
                    features['embedding'],
                    features['confidence'],
                    features['gradient'],
                )
            )

    mir_metrics = run_full_metrics(
        prediction_cache,
        dataset,
        anchors,
        conf_threshold,
        nms_iou_threshold,
    )
    metrics.update({
        f"{prefix}_P": mir_metrics["P"],
        f"{prefix}_R": mir_metrics["R"],
        f"{prefix}_F1": mir_metrics["F1"],
        f"{prefix}_AOR": mir_metrics["AOR"],
        f"{prefix}_P_op": mir_metrics["P_op"],
        f"{prefix}_R_op": mir_metrics["R_op"],
        f"{prefix}_F1_op": mir_metrics["F1_op"],
    })
    return metrics

def train(cfg: dict | None = None, resume: bool = False):
    # --- 1. Hardware Setup ---
    cfg = cfg or {}
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})
    train_cfg = cfg.get("train", {})
    mlflow_cfg = cfg.get("mlflow", {})
    eval_cfg = cfg.get("eval", {})

    requested_device = train_cfg.get(
        "device",
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(requested_device)
    print(f"--- Training Session Started ---")
    print(f"Target Device: {device}")

    # --- 2. Configuration ---
    # Legacy fallback values kept here for quick rollback after the smoke test:
    # checkpoint_dir = 'checkpoints'
    # processed_dir = 'data/processed/itm_flute'
    # batch_size = 4
    # num_epochs = 150
    # learning_rate = 1e-4
    # weight_decay = 1e-4
    # num_workers = 0
    # grad_clip = 5.0
    # save_every = 10

    output_root = cfg.get("output_root", "outputs")
    run_name = cfg.get("run_name", "legacy_run")
    checkpoint_dir = os.path.join(output_root, run_name, "checkpoints")
    processed_dir = data_cfg.get("processed_dir", 'data/processed/itm_flute')
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 0)
    train_size = data_cfg.get("train_size", 0.64)
    val_size = data_cfg.get("val_size", 0.16)
    test_size = data_cfg.get("test_size", 0.20)
    combine_val_to_train = data_cfg.get("combine_val_to_train", False)
    train_set_usage = data_cfg.get("train_set_usage", 1.0)
    num_epochs = train_cfg.get("epochs", 150)
    learning_rate = optim_cfg.get("lr", 1e-4)
    weight_decay = optim_cfg.get("weight_decay", 1e-4)
    grad_clip = train_cfg.get("grad_clip", 5.0)
    save_every = train_cfg.get("save_every", 10)
    train_metrics_max_samples = train_cfg.get("train_metrics_max_samples", 32)
    num_anchors = model_cfg.get("num_anchors", 3)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Processed data directory: {processed_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # --- 3. Data Preparation ---
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Data directory {processed_dir} not found. Run preprocess_dataset.py and precompute_features.py first.")
        
    all_stems = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    train_stems, val_stems, test_stems = get_train_val_test_split(
        all_stems,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        combine_val_to_train=combine_val_to_train,
        train_set_usage=train_set_usage,
        seed=cfg.get("seed", 42),
    )
    print(
        f"Dataset Split: {len(train_stems)} training | "
        f"{len(val_stems)} validation | {len(test_stems)} test"
    )
    print(
        f"Split Mode: combine_val_to_train={combine_val_to_train} | "
        f"train_set_usage={train_set_usage:.2f}"
    )

    train_dataset = MusicNoteDataset(processed_dir=processed_dir, stems=train_stems)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=music_yolo_collate_fn,
        num_workers=num_workers
    )
    val_dataset = MusicNoteDataset(processed_dir=processed_dir, stems=val_stems)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=music_yolo_collate_fn,
        num_workers=num_workers,
    )
    test_dataset = MusicNoteDataset(processed_dir=processed_dir, stems=test_stems)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=music_yolo_collate_fn,
        num_workers=num_workers,
    )
    train_metric_dataset = build_eval_subset(train_dataset, train_metrics_max_samples)
    train_metric_dataloader = DataLoader(
        train_metric_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=music_yolo_collate_fn,
        num_workers=num_workers,
    )

    # --- 4. Model & Optimizer Initialization ---
    model = DualStreamMusicYOLO(num_anchors=num_anchors).to(device)
    loss_fn = MusicYOLOLoss(num_anchors=num_anchors).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # OneCycleLR automatically handles the warmup ramp-up, then the cosine decay down
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4, 
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.05 # Dedicate the first 5% of training to the Warmup Phase
    )

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
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        active_run_id = checkpoint.get('run_id') # Retrieve the original MLflow run ID
        print(f"✅ Successfully loaded. Restarting at Epoch {start_epoch}")
    else:
        reason = "Resume not requested" if not resume else "No previous checkpoints found"
        print(f"🌱 Starting fresh training run. ({reason})")

    # --- 6. MLflow Tracking ---
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "CREPE-YOLO-Transcription"))
    
    # If active_run_id is None, a new run starts. If it exists, it re-opens the old one.
    with mlflow.start_run(run_id=active_run_id) as run:
        current_run_id = run.info.run_id
        best_val_f1_op = float("-inf")
        best_val_loss = float("inf")
        best_ckpt_path = os.path.join(checkpoint_dir, "best_val_f1_op.pt")
        if active_run_id:
            print(f"📈 Continuing MLflow Run: {current_run_id}")
        else:
            print(f"🚀 New MLflow Run ID: {current_run_id}")
            mlflow.log_params({
                "run_name": run_name,
                "processed_dir": processed_dir,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "save_every": save_every,
                "num_anchors": num_anchors,
                "train_samples": len(train_stems),
                "val_samples": len(val_stems),
                "test_samples": len(test_stems),
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "combine_val_to_train": combine_val_to_train,
                "train_set_usage": train_set_usage,
                "train_metrics_max_samples": min(train_metrics_max_samples, len(train_dataset)),
                "device": str(device)
            })
        print(f"Checkpoint directory: {checkpoint_dir}")

        best_test_f1_op = float("-inf")
        best_test_ckpt_path = os.path.join(checkpoint_dir, "best_test_f1_op.pt")
        split_metadata = {
            "train_samples": len(train_stems),
            "val_samples": len(val_stems),
            "test_samples": len(test_stems),
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "combine_val_to_train": combine_val_to_train,
            "train_set_usage": train_set_usage,
        }

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

                loss_dict = loss_fn(predictions, targets)
                loss = loss_dict['total_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()

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
            epoch_metrics = {
                "train_loss": avg_loss,
                "loss_objectness": epoch_obj_loss / len(dataloader),
                "loss_box_coord": epoch_box_loss / len(dataloader)
            }
            train_metrics = evaluate_split(
                model,
                loss_fn,
                train_metric_dataloader,
                train_metric_dataset,
                device,
                eval_cfg,
                "train_eval",
            )
            val_metrics = evaluate_split(
                model,
                loss_fn,
                val_dataloader,
                val_dataset,
                device,
                eval_cfg,
                "val",
            )
            test_metrics = evaluate_split(
                model,
                loss_fn,
                test_dataloader,
                test_dataset,
                device,
                eval_cfg,
                "test",
            )
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(val_metrics)
            epoch_metrics.update(test_metrics)
            mlflow.log_metrics(epoch_metrics, step=epoch + 1)

            if "val_F1_op" in val_metrics and val_metrics["val_F1_op"] > best_val_f1_op:
                best_val_f1_op = val_metrics["val_F1_op"]
                torch.save({
                    'epoch': epoch,
                    'run_id': current_run_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'split_metadata': split_metadata,
                    'best_val_F1_op': best_val_f1_op,
                    'test_F1_op_at_best_val': test_metrics.get("test_F1_op"),
                }, best_ckpt_path)
                print(f"🏆 Updated best validation model: {best_ckpt_path}")

            if "test_F1_op" in test_metrics and test_metrics["test_F1_op"] > best_test_f1_op:
                best_test_f1_op = test_metrics["test_F1_op"]
                torch.save({
                    'epoch': epoch,
                    'run_id': current_run_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'split_metadata': split_metadata,
                    'best_test_F1_op': best_test_f1_op,
                    'val_F1_op_at_best_test': val_metrics.get("val_F1_op"),
                }, best_test_ckpt_path)
                print(f"🧪 Updated best test model: {best_test_ckpt_path}")

            if "val_loss" in val_metrics:
                best_val_loss = min(best_val_loss, val_metrics["val_loss"])

            print(
                f"End of Epoch {epoch+1} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Eval F1(op): {train_metrics.get('train_eval_F1_op', float('nan')):.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', float('nan')):.4f} | "
                f"Val F1(op): {val_metrics.get('val_F1_op', float('nan')):.4f} | "
                f"Test Loss: {test_metrics.get('test_loss', float('nan')):.4f} | "
                f"Test F1(op): {test_metrics.get('test_F1_op', float('nan')):.4f}"
            )

            # --- 8. Periodic Checkpointing ---
            if save_every > 0 and (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"crepe_yolo_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'run_id': current_run_id, # CRITICAL: Saves the run ID so charts connect
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'shceduler_state_dict': scheduler.state_dict(),
                    'config': cfg,
                    'split_metadata': split_metadata,
                }, ckpt_path)
                
                mlflow.log_artifact(ckpt_path)
                print(f"💾 Saved checkpoint to {ckpt_path}")

        if best_val_f1_op != float("-inf"):
            mlflow.log_metric("best_val_F1_op", best_val_f1_op)
            if os.path.exists(best_ckpt_path):
                mlflow.log_artifact(best_ckpt_path)
        if best_test_f1_op != float("-inf"):
            mlflow.log_metric("best_test_F1_op", best_test_f1_op)
            if os.path.exists(best_test_ckpt_path):
                mlflow.log_artifact(best_test_ckpt_path)
        if best_val_loss != float("inf"):
            mlflow.log_metric("best_val_loss", best_val_loss)

    print("--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Resume CREPE-YOLO")
    parser.add_argument('--resume', action='store_true', help="Try to resume from the latest checkpoint if it exists")
    parser.add_argument('--config', type=str, default=None, help="Optional YAML config path.")
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else None
    train(cfg=cfg, resume=args.resume)
