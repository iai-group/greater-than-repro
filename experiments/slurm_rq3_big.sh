#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --job-name=rq3gate
#SBATCH --output=rq3gate.out
#SBATCH --error=rq3gate.err

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

# Gate first: a model that cannot do the task at all must not reach discovery.
# Restricted to the two new checkpoints: the other six already have committed
# results, and re-running them only risks replacing good entries with OOMs.
# Gate only: discovery already completed for both new checkpoints. The gate
# merges, so re-running it repairs pythia-12b's OOMed entry without touching
# the seven that succeeded.
uv run python -u experiments/rq3_crossmodel.py --gate 2>&1 | tee rq3big_gate.log

echo "Finished: $(date)"
