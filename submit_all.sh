#!/usr/bin/env bash
set -euo pipefail

# Submits the SLURM scale-up pipeline in dependency order:
#   1) extract_embeddings.sbatch  (job array over the full dataset)
#   2) finetune_encoders.sbatch   (waits for (1) to finish successfully)
# Not executed as part of this repo — see slurm/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

EXTRACT_JOB_ID="$(sbatch --parsable extract_embeddings.sbatch)"
echo "Submitted extract_embeddings.sbatch -> job ${EXTRACT_JOB_ID}"

FINETUNE_JOB_ID="$(sbatch --parsable --dependency=afterok:"${EXTRACT_JOB_ID}" finetune_encoders.sbatch)"
echo "Submitted finetune_encoders.sbatch -> job ${FINETUNE_JOB_ID} (dependency: afterok:${EXTRACT_JOB_ID})"

echo "Monitor with: squeue -u \$USER"
