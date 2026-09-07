# Runs the full multimodal price-prediction pipeline end to end, in order.
#
# Usage:
#   ./scripts/run_pipeline.ps1
#   $env:PYTHON = "C:\path\to\venv\python.exe"; ./scripts/run_pipeline.ps1
#
# Notes:
#   - Image embedding extraction here processes the whole subset in a single
#     call. On a slow CPU this can take a while (see reports/RUN_NOTES.md for
#     this project's measured throughput). If you need to stay under a
#     command-timeout budget, chunk it manually instead, e.g.:
#       & $Python src/extract_embeddings.py --modality image --start 0    --end 2000
#       & $Python src/extract_embeddings.py --modality image --start 2000 --end 4000
#       ...
#       & $Python src/extract_embeddings.py --modality image --merge
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Set-Location $RepoRoot

Write-Host "=== [1/5] prepare_data ==="
& $Python src/prepare_data.py
if ($LASTEXITCODE -ne 0) { throw "prepare_data.py failed" }

Write-Host "=== [2/5] extract_embeddings (text) ==="
& $Python src/extract_embeddings.py --modality text
if ($LASTEXITCODE -ne 0) { throw "extract_embeddings.py (text) failed" }

Write-Host "=== [3/5] extract_embeddings (image) ==="
& $Python src/extract_embeddings.py --modality image
if ($LASTEXITCODE -ne 0) { throw "extract_embeddings.py (image) failed" }

Write-Host "=== [4/5] train ==="
& $Python src/train.py
if ($LASTEXITCODE -ne 0) { throw "train.py failed" }

Write-Host "=== [5/5] evaluate ==="
& $Python src/evaluate.py
if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed" }

Write-Host "Pipeline complete. See reports/metrics.json and reports/figures/."
