#!/usr/bin/env bash
set -euo pipefail

# Syncs this repository to the SLURM cluster for the full-dataset GPU path
# (see slurm/README.md). Ships code/config/docs and a lightweight data
# manifest; does NOT ship the raw dataset or generated artifacts — those
# are expected to already live on cluster-visible storage.

REMOTE_USER="vintia"
REMOTE_HOST="172.24.16.132"
REMOTE_PATH="/nfs_home/users/vintia/Tanay/personal/Multimodal_Price_Prediction"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Syncing ${PROJECT_ROOT} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

rsync -avz --progress \
  --exclude='.git/' \
  --exclude='artifacts/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='data/dataset/' \
  --exclude='data/Train_Images/' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='*.safetensors' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='slurm/logs/' \
  "${PROJECT_ROOT}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

# Ship a lightweight manifest of the local dataset (row count + column
# names only, no raw content) so the remote side can sanity-check it has a
# matching copy already staged.
LOCAL_TRAIN_CSV="${PROJECT_ROOT}/data/dataset/train.csv"
if [ -f "${LOCAL_TRAIN_CSV}" ]; then
  MANIFEST="$(mktemp)"
  {
    echo "synced_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "rows: $(($(wc -l < "${LOCAL_TRAIN_CSV}") - 1))"
    echo "columns: $(head -n1 "${LOCAL_TRAIN_CSV}")"
  } > "${MANIFEST}"
  scp "${MANIFEST}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/MANIFEST.txt"
  rm -f "${MANIFEST}"
  echo "Wrote data/MANIFEST.txt on remote."
else
  echo "No local data/dataset/train.csv found — skipping manifest (code/config synced only)."
fi

echo "Done."
