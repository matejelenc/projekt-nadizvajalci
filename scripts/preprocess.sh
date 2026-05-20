#!/usr/bin/env bash
#SBATCH --job-name=preprocess
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%j_stdout.log
#SBATCH --error=logs/slurm/%j_stderr.log

set -euo pipefail

# Resolve repo root relative to this script's location.
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module load CUDA/12.6.0

source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"
python -u -m geonews.nlp.preprocess
