#!/bin/bash
# Train CHROMA-Net on all three PanNuke splits as a job array, one split per task.
#
# The three splits are independent, so they run concurrently rather than in
# sequence. Every published PanNuke number is the average over all three, so
# there is no point running only one.
#
# Usage:
#     sbatch scripts/train_chroma.sbatch
#     sbatch --export=ENCODER=uni,TAG=chroma_uni scripts/train_chroma.sbatch
#
#SBATCH --job-name=chroma
#SBATCH --array=1-3
#SBATCH --partition=rli2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=logs/chroma_%A_%a.out
#SBATCH --error=logs/chroma_%A_%a.err

set -euo pipefail

ENCODER=${ENCODER:-phikon}
TAG=${TAG:-chroma}
EPOCHS=${EPOCHS:-50}
# 8 fits a 24GB card with consistency=teacher, which runs three ViT-L
# forward passes per step. Raise to 16 on a 40GB+ GPU:
#     sbatch --export=ALL,BATCH_SIZE=16 scripts/train_chroma.sbatch
BATCH_SIZE=${BATCH_SIZE:-8}
EXTRA=${EXTRA:-}

mkdir -p logs

echo "host        $(hostname)"
echo "split       ${SLURM_ARRAY_TASK_ID}"
echo "encoder     ${ENCODER}"
echo "tag         ${TAG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Ignore ~/.local/lib/python*/site-packages. A stray package there shadows the
# conda env on every node, and a broken one takes the job with it: a partial
# scikit-learn in ~/.local made transformers' optional sklearn probe succeed
# and then fail on its missing joblib, killing model construction. Jobs should
# see only what the environment declares.
export PYTHONNOUSERSITE=1

# Keep the model cache on scratch rather than $HOME, which is quota-limited on
# most clusters and will silently fail a 1.2 GB ViT-L download.
export HF_HOME=${HF_HOME:-${SCRATCH:-/tmp}/huggingface}
export TOKENIZERS_PARALLELISM=false

# Compute nodes here have no outbound internet, so a cold cache would stall
# until the connection times out rather than failing. Offline mode turns
# that into an immediate, legible error. Warm the cache first with
# scripts/prefetch_encoder.sh on a login node.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

python src/train_chroma.py \
    --split "${SLURM_ARRAY_TASK_ID}" \
    --encoder "${ENCODER}" \
    --tag "${TAG}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    ${EXTRA}
