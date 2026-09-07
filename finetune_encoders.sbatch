#!/bin/bash
#SBATCH --job-name=mpp-finetune-encoders
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/finetune_%j.out
#SBATCH --error=slurm/logs/finetune_%j.err

# Scripted scale-up path — NOT executed. LoRA-finetunes the ViT-S/16 image
# tower and MiniLM-L6-v2 text tower jointly with the MLP fusion head on the
# full ~75k-row dataset, on a single GPU. Every result in README.md and
# docs/EXPERIMENTS.md comes from the frozen-encoder CPU subsample path
# (scripts/run_pipeline.sh); this script documents the intended full-scale
# finetuning path and has not been run.
#
# Requires `peft` (LoRA) in addition to requirements.txt — install into the
# same venv used by extract_embeddings.sbatch before submitting.

set -euo pipefail

PROJECT_ROOT="/nfs_home/users/vintia/Tanay/personal/Multimodal_Price_Prediction"
VENV_PATH="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

mkdir -p artifacts/checkpoints/finetuned slurm/logs

echo "[job ${SLURM_JOB_ID:-local}] host=$(hostname) starting encoder finetuning"

python src/finetune_encoders.py \
  --config config/config.yaml \
  --data-root data/ \
  --full \
  --image-backbone vit_small_patch16_224.augreg_in21k_ft_in1k \
  --text-backbone sentence-transformers/all-MiniLM-L6-v2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules qkv,proj,query,key,value,dense \
  --epochs 5 \
  --batch-size 64 \
  --lr 2e-4 \
  --warmup-ratio 0.03 \
  --precision bf16 \
  --grad-accum-steps 4 \
  --output-dir artifacts/checkpoints/finetuned/ \
  --device cuda \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 42

echo "[job ${SLURM_JOB_ID:-local}] finetune_encoders complete"
