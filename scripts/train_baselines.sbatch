#!/bin/bash
# Train the binary baselines on all three splits.
#
# Nine runs (three models x three splits) laid out as a job array. Task index
# maps to (model, split) so each gets its own GPU rather than queueing behind
# the others.
#
# Usage:
#     sbatch scripts/train_baselines.sbatch
#     sbatch --export=EXTRA=--no-augment scripts/train_baselines.sbatch
#
#SBATCH --job-name=baselines
#SBATCH --array=0-8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err

set -euo pipefail

MODELS=(cnn unet attention_unet)
EPOCHS=${EPOCHS:-50}
EXTRA=${EXTRA:-}

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SPLIT=$((SLURM_ARRAY_TASK_ID % 3 + 1))

mkdir -p logs

echo "host        $(hostname)"
echo "model       ${MODEL}"
echo "split       ${SPLIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --epochs "${EPOCHS}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    ${EXTRA}
