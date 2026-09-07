#!/bin/bash
#SBATCH --job-name=mpp-extract-embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm/logs/extract_%A_%a.out
#SBATCH --error=slurm/logs/extract_%A_%a.err

# Full-dataset embedding extraction — scripted scale-up path, NOT executed
# for this repo's reported results. Embeds all ~75k rows with the frozen
# ViT-S/16 + MiniLM-L6-v2 towers used by the CPU pipeline, split into
# SLURM_ARRAY_TASK_COUNT chunks so it parallelizes across array tasks.
# Every number in README.md / docs/EXPERIMENTS.md comes from the CPU
# subsample run in scripts/run_pipeline.sh instead.

set -euo pipefail

PROJECT_ROOT="/nfs_home/users/vintia/Tanay/personal/Multimodal_Price_Prediction"
VENV_PATH="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

N_CHUNKS="${SLURM_ARRAY_TASK_COUNT:-10}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p artifacts/embeddings/full slurm/logs

echo "[job ${SLURM_JOB_ID:-local} / array task ${CHUNK_ID}] host=$(hostname) chunk ${CHUNK_ID} of ${N_CHUNKS}"

python src/extract_embeddings.py \
  --config config/config.yaml \
  --data-root data/ \
  --full \
  --n-chunks "${N_CHUNKS}" \
  --chunk-id "${CHUNK_ID}" \
  --output-dir artifacts/embeddings/full/ \
  --device cuda \
  --batch-size 256 \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"

echo "[chunk ${CHUNK_ID}/${N_CHUNKS}] done"
