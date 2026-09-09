#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --job-name=gt-tl
#SBATCH --output=gt-tl.out
#SBATCH --error=gt-tl.err

set -euo pipefail

# Set REPO_DIR to this repository's path on your cluster before submitting.
REPO_DIR="${REPO_DIR:?set REPO_DIR, e.g. REPO_DIR=/path/to/greater-than-repro sbatch $0}"
cd "$REPO_DIR"

# --- uv: env is pre-synced on the LOGIN node; the GPU node has no internet, so
#     forbid any implicit sync/download here (mirrors the rust_circuit setup).
export UV_LINK_MODE=copy
export UV_NO_SYNC=1

# --- HuggingFace: gpt2 weights must be PRE-CACHED on the login node (see README
#     steps); force offline so the node never tries to reach huggingface.co.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- runtime niceties
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then export CUDA_VISIBLE_DEVICES=0; fi

echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "HF_HOME: $HF_HOME"
echo "============================================================"
nvidia-smi || true

# RQ1 baseline reproduction (gpt2-small). For RQ3 later, rerun with --model
# EleutherAI/pythia-410m, meta-llama/Meta-Llama-3-8B, etc.
uv run python -u reproduce_greater_than.py --model gpt2-small 2>&1 | tee gt-tl.log

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
