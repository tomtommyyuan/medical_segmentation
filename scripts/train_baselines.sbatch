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
#SBATCH --partition=rli2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err

set -euo pipefail

MODELS=(cnn unet attention_unet)
EPOCHS=${EPOCHS:-50}
EXTRA=${EXTRA:-}

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SPLIT=$((SLURM_ARRAY_TASK_ID % 3 + 1))

# Ignore ~/.local/lib/python*/site-packages. A stray package there shadows the
# conda env on every node, and a broken one takes the job with it: a partial
# scikit-learn in ~/.local made transformers' optional sklearn probe succeed
# and then fail on its missing joblib, killing model construction. Jobs should
# see only what the environment declares.
export PYTHONNOUSERSITE=1

# Slurm redirects stdout to a file, which Python block-buffers rather than
# line-buffers, so epoch lines sit in an 8KB buffer and a healthy job looks
# stalled for its first half hour. The history CSV is flushed explicitly and
# stays accurate either way, but the log should be readable live.
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "host        $(hostname)"
echo "model       ${MODEL}"
echo "split       ${SPLIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate the environment explicitly rather than relying on --export=ALL to
# carry the submitting shell's PATH. A resubmission from a shell where the env
# is not active would otherwise run the base interpreter and die on the first
# import, several minutes into the job.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-medseg}"

echo "python      $(which python)"

python src/train.py \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --epochs "${EPOCHS}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    ${EXTRA}
