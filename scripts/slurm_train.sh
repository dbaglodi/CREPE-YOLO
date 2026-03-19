#!/bin/bash
#SBATCH -J crepe_train_smoke
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lalexeyev3@gatech.edu
#SBATCH --export=NONE

set -euo pipefail

CONFIG_PATH="${1:-configs/test.yaml}"
RESUME_FLAG="${RESUME_FLAG:-}"

module load anaconda3

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate /storage/ice1/1/5/lalexeyev3/conda/crepe-yolo

cd /storage/ice1/1/5/lalexeyev3/CREPE-YOLO

python scripts/run_train.py --config "${CONFIG_PATH}" ${RESUME_FLAG}
