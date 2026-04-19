# CREPE-YOLO Transcription Project

CREPE-YOLO trains a YOLO-style note detector on top of precomputed CREPE features for monophonic transcription experiments. This README focuses on the repo's current training workflow: preparing features, running a single training job, launching a hyperparameter sweep, selecting the best run, and understanding how YAML configs control behavior.

## Setup

Create the environment from the repo config:

```bash
conda env create -f configs/environment.yml
conda activate crepeyolo
```

Optional install check:

```bash
python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))"
```

## Workflow Overview

The intended workflow is:

1. Prepare the dataset and precompute CREPE features.
2. Run a single training job to verify the pipeline.
3. Launch a hyperparameter sweep when you want to tune training settings.
4. Summarize sweep logs and choose the best run by validation `F1(op)`.
5. Optionally retrain the best settings across different train-set sizes.

TensorFlow and PyTorch are intentionally separated across scripts. Feature extraction happens in training/precompute_features.py, while training happens in scripts/run_train.py. Keep that separation in place to avoid framework allocator conflicts.

## Data Preparation

If you are starting from the raw ITM-Flute dataset, preprocess it first:

```bash
python training/preprocess_dataset.py
python training/precompute_features.py
```

Defaults baked into the scripts today:

- `training/preprocess_dataset.py` expects the raw dataset under `./GT-ITM-Flute-99` and writes processed stems to `./data/processed/itm_flute`.
- `training/precompute_features.py` reads from `./data/processed/itm_flute` and creates `features.pt` tensors for each stem.

If `data/processed/itm_flute` already exists and contains `features.pt` files, you can skip this step.

## Run One Training Job

Use scripts/run_train.py for all single-run training jobs:

```bash
python scripts/run_train.py --config configs/base.yaml
```

Use the smoke-test config when you want a short end-to-end run:

```bash
python scripts/run_train.py --config configs/test.yaml
```

Resume from the latest checkpoint in the run directory:

```bash
python scripts/run_train.py --config configs/base.yaml --resume
```

What each command is for:

- `configs/base.yaml`: normal training run.
- `configs/test.yaml`: quick dry run with very few epochs.
- `--resume`: reload the latest checkpoint in `outputs/<run_name>/checkpoints/` and continue the same MLflow run when possible.

Training artifacts are written under:

```text
outputs/<run_name>/
```

Important files inside that directory:

- `resolved_config.yaml`: the exact config snapshot used for the run.
- `checkpoints/`: periodic checkpoints plus best-model checkpoints.
- MLflow metrics and params: tracked via the configured MLflow experiment.

To inspect training curves locally:

```bash
mlflow ui
```

In order to submit this to pace, use slurm_sweep.sh and adjust parameters at the top.

## Hyperparameter Tuning Loop

Use scripts/launch_sweep.py to generate a family of configs from a base YAML:

```bash
python scripts/launch_sweep.py --base-config configs/sweep.yaml --output-dir generated_sweeps
```

That command creates one YAML per hyperparameter combination plus a manifest:

```text
generated_sweeps/configs.txt
```

To submit the generated configs as a SLURM array:

```bash
python scripts/launch_sweep.py --base-config configs/sweep.yaml --output-dir generated_sweeps --submit
```

Current sweep dimensions in `launch_sweep.py`:

- `optim.lr`
- `data.batch_size`
- `optim.weight_decay`
- `train.grad_clip`

The sweep launcher also overrides a few run-shaping fields from CLI arguments:

- `data.train_size`
- `data.val_size`
- `data.test_size`
- `data.combine_val_to_train=false`
- `data.train_set_usage=1.0`
- `train.save_every`

How the SLURM flow works:

- `launch_sweep.py` writes all generated config paths into `configs.txt`.
- scripts/slurm_sweep.py reads that manifest.
- Each array task selects one line from the manifest and runs `python scripts/run_train.py --config <generated-config>`.

If you only want to inspect or edit the generated configs before submitting, stop after the non-`--submit` command and look in `generated_sweeps/`.

## Summarize And Pick The Best Sweep Run

After the sweep completes, summarize the logs with scripts/summarize_sweep.py:

```bash
python scripts/summarize_sweep.py --logs-glob 'logs/crepe_sweep_a40_<jobid>_*.out'
```

The script ranks runs by best validation `F1(op)` and prints the top configurations. The two most important outputs are:

- the recommended `resolved_config.yaml` path for the best run
- the best checkpoint path for that run

Useful optional flags:

```bash
python scripts/summarize_sweep.py --logs-glob 'logs/crepe_sweep_a40_<jobid>_*.out' --top-k 5
python scripts/summarize_sweep.py --logs-glob 'logs/crepe_sweep_a40_<jobid>_*.out' --json summaries/sweep.json
```

## Retrain Best Hyperparameters Across Train-Set Sizes

Use scripts/launch_train_size_sweep.py when you want to keep the best hyperparameters fixed and vary how much training data is used.

You can point it directly at a resolved config:

```bash
python scripts/launch_train_size_sweep.py --base-config <resolved-config-path>
```

Or let it infer the best config from sweep logs:

```bash
python scripts/launch_train_size_sweep.py --logs-glob 'logs/crepe_sweep_a40_<jobid>_*.out'
```

This script modifies the selected config by:

- setting `data.combine_val_to_train=true`
- sweeping `data.train_set_usage`
- preserving the chosen training hyperparameters

Like the main sweep launcher, it writes generated configs plus a `configs.txt` manifest and can submit through the same SLURM array wrapper with `--submit`.

## How Configs Work

All training runs start from a YAML file such as configs/base.yaml, configs/test.yaml, or configs/sweep.yaml.

### Top-level keys

- `seed`: global random seed used for deterministic splitting and training setup.
- `run_name`: name of the run directory under `output_root`.
- `output_root`: parent directory for training artifacts.
- `data`: dataset path, dataloader settings, split fractions, and train-size controls.
- `model`: model architecture settings such as `num_anchors`.
- `optim`: optimizer settings such as learning rate and weight decay.
- `train`: training-time settings such as device, epochs, gradient clipping, checkpoint cadence, and train-metric subsampling.
- `mlflow`: tracking URI and experiment name.
- `eval`: confidence threshold, NMS IoU threshold, and MIR-style evaluation tolerances.

### `data.*`

`data.processed_dir` points to the precomputed feature directory. `data.batch_size` and `data.num_workers` control the dataloader. `data.train_size`, `data.val_size`, and `data.test_size` define the split fractions. `data.combine_val_to_train` and `data.train_set_usage` are used for retraining or utilization studies after hyperparameter selection.

### `model.*`

`model.num_anchors` controls the YOLO head anchor count used by the model and loss.

### `optim.*`

`optim.name` records the intended optimizer family, and the training code currently uses AdamW with `optim.lr` and `optim.weight_decay`.

### `train.*`

`train.epochs` sets the training length. `train.device` requests `cuda`, `mps`, or `cpu`, and the code falls back if the requested backend is unavailable. `train.grad_clip`, `train.save_every`, and `train.train_metrics_max_samples` control optimization stability, periodic checkpointing, and how much of the training set is evaluated each epoch.

### `mlflow.*`

`mlflow.tracking_uri` and `mlflow.experiment_name` determine where run metadata is logged.

### `eval.*`

`eval.conf_threshold`, `eval.nms_iou_threshold`, and the MIR evaluation tolerances control how predictions are decoded and scored during validation and testing.

`scripts/run_train.py` always snapshots the loaded config to:

```text
outputs/<run_name>/resolved_config.yaml
```

Treat that file as the config-of-record for the run.

## Config Patterns

Recommended starting points:

- Start from `configs/test.yaml` for smoke tests and environment checks.
- Use `configs/base.yaml` for a normal training run.
- Use `configs/sweep.yaml` as the base template for hyperparameter tuning.

The sweep scripts do not introduce a separate config system. They load a normal YAML config, overwrite a targeted subset of fields, and save each modified variant back out as another ordinary YAML file.

## Outputs And Naming

`run_name` controls the output folder name under `output_root`, so choose a name that makes the experiment easy to identify later.

For a typical run, expect to see:

- `outputs/<run_name>/resolved_config.yaml`
- `outputs/<run_name>/checkpoints/crepe_yolo_epoch_<N>.pt` for periodic saves
- `outputs/<run_name>/checkpoints/best_val_f1_op.pt`
- `outputs/<run_name>/checkpoints/best_test_f1_op.pt`
- MLflow metrics including best validation and test `F1(op)`

For sweep jobs, also expect:

- generated config files under the chosen sweep output directory
- a `configs.txt` manifest listing those generated configs
- SLURM log files under `logs/`

## Evaluation Notes

Evaluation code lives in training/evaluate.py. It handles decoding, NMS, and MIR-style transcription metrics. For day-to-day training and sweep selection, the main number to watch is validation `F1(op)`, since that is also what the sweep summarizer uses for ranking.
