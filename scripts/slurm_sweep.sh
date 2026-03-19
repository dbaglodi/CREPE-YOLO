#!/bin/bash
#SBATCH -J crepe_sweep_a40
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lalexeyev3@gatech.edu
#SBATCH --export=NONE

set -euo pipefail

MANIFEST_PATH="${1:?Usage: sbatch scripts/slurm_sweep.sh <manifest-path>}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load anaconda3

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate /storage/ice1/1/5/lalexeyev3/conda/crepe-yolo

cd /storage/ice1/1/5/lalexeyev3/CREPE-YOLO

CONFIG_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST_PATH}")"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "No config found for array task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "Running sweep task ${SLURM_ARRAY_TASK_ID} with config: ${CONFIG_PATH}"
python scripts/run_train.py --config "${CONFIG_PATH}"
