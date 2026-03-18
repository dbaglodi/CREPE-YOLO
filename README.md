# CREPE-YOLO Transcription Project
## Quick Start (Linux/CUDA or Mac/MPS)
Setup Env: conda env create -f environment.yml

Activate: conda activate crepeyolo

## Verify Device: 
Run `python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))"`

## Architecture & Framework "Airlock"
This project uses TensorFlow (for CREPE embeddings) and PyTorch (for the YOLO head). These two frameworks often collide over GPU memory allocators.

DO NOT import both frameworks in the same script.

## Pre-computation:
Run python precompute_features.py. This script uses a Subprocess Airlock to isolate TensorFlow during feature extraction before passing the data to PyTorch. This is required for stability on both Linux and macOS.

## Training & Evaluation
MLflow: Training is tracked via MLflow. Run mlflow ui in the project root to see the loss curves.

## Evaluation:
evaluate.py handles the MIR-eval metrics. It automatically handles the NMS operation on the CPU to ensure compatibility across CUDA and MPS backends.