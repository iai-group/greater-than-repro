#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem=200G
#SBATCH --job-name=rq4qwn
#SBATCH --output=rq4qwn2.out
#SBATCH --error=rq4qwn2.err

set -euo pipefail

# Set REPO_DIR to this repository's path on your cluster before submitting.
REPO_DIR="${REPO_DIR:?set REPO_DIR, e.g. REPO_DIR=/path/to/greater-than-repro sbatch $0}"
cd "$REPO_DIR"

# SLURM batch scripts do not source the login profile, so uv (installed
# under ~/.local/bin) is not on PATH and the job dies with exit 127.
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
export UV_NO_SYNC=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then export CUDA_VISIBLE_DEVICES=0; fi

echo "Job started: $(date) on $(hostname)"
nvidia-smi || true

# Only the top budget: the smaller ones completed in the prior run and
# results merge, so this recovers the point that hit a CUDA error.
uv run python -u experiments/rq4_qwen_normalized.py --fractions 0.012 2>&1 | tee rq4qwn2.log

echo "Finished: $(date)"
