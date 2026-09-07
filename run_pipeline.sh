#!/usr/bin/env bash
# Runs the full multimodal price-prediction pipeline end to end, in order.
#
# Usage:
#   ./scripts/run_pipeline.sh
#   PYTHON=/path/to/venv/python ./scripts/run_pipeline.sh
#
# Notes:
#   - Image embedding extraction here processes the whole subset in a single
#     call. On a slow CPU this can take a while (see reports/RUN_NOTES.md for
#     this project's measured throughput). If you need to stay under a
#     command-timeout budget, chunk it manually instead, e.g.:
#       $PYTHON src/extract_embeddings.py --modality image --start 0    --end 2000
#       $PYTHON src/extract_embeddings.py --modality image --start 2000 --end 4000
#       ...
#       $PYTHON src/extract_embeddings.py --modality image --merge
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$REPO_ROOT"

echo "=== [1/5] prepare_data ==="
"$PYTHON" src/prepare_data.py

echo "=== [2/5] extract_embeddings (text) ==="
"$PYTHON" src/extract_embeddings.py --modality text

echo "=== [3/5] extract_embeddings (image) ==="
"$PYTHON" src/extract_embeddings.py --modality image

echo "=== [4/5] train ==="
"$PYTHON" src/train.py

echo "=== [5/5] evaluate ==="
"$PYTHON" src/evaluate.py

echo "Pipeline complete. See reports/metrics.json and reports/figures/."
