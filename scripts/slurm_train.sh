#!/bin/bash
#SBATCH -J crepe_yolo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C 'gpu-a100'
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lalexeyev3@gatech.edu

# load modules
module load anaconda3

# activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc.profile.d/conda.sh
conda activate /storage/ice1/1/5/lalexeyev3/conda/crepe-yolo

# Verify environment setup
echo "=== Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo ""
echo "=== PyTorch Check ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>&1 || echo "ERROR: Could not import PyTorch - check installation and library paths"
echo ""

cd /storage/ice1/1/5/lalexeyev3/crepe-yolo

python scripts/run_train.py --config configs/base.yaml
