# SLURM scale-up path

This directory scripts the **full-dataset GPU path** described in the main [README](../README.md#scaling-up-gpuslurm): embedding extraction over the complete ~75k-row dataset, and LoRA-finetuning both encoder towers. Everything here is written to be correct and ready to submit against the target cluster, but **has not been executed** — every number in this repository's README/EXPERIMENTS.md comes from the CPU subsample path in `scripts/`, not from here. See [`docs/PROGRESS.md`](../docs/PROGRESS.md) for status.

## Layout

| File | Purpose |
|---|---|
| `sync_to_cluster.sh` | rsyncs the repository (code + config + a data manifest, not the raw images) to the cluster |
| `extract_embeddings.sbatch` | job-array sbatch: embeds the full dataset with the frozen ViT + MiniLM towers, chunked across array tasks |
| `finetune_encoders.sbatch` | sbatch skeleton: LoRA-finetunes the ViT and MiniLM towers jointly with the MLP fusion head on the full dataset |
| `submit_all.sh` | submits both jobs in dependency order (`finetune_encoders` waits on `extract_embeddings`) |

## Usage

From a machine with SSH access to the cluster:

```bash
# 1. push the repo (code/config, not raw data) to the cluster
./slurm/sync_to_cluster.sh

# 2. on the cluster, from the project root:
ssh vintia@172.24.16.132
cd /nfs_home/users/vintia/Tanay/personal/Multimodal_Price_Prediction
./slurm/submit_all.sh                  # submits extract_embeddings then finetune_encoders
# ...or individually:
sbatch slurm/extract_embeddings.sbatch
sbatch slurm/finetune_encoders.sbatch

# 3. monitor
squeue -u $USER

# 4. once complete, pull results back (metrics/figures only — see .gitignore)
rsync -avz vintia@172.24.16.132:/nfs_home/users/vintia/Tanay/personal/Multimodal_Price_Prediction/reports/ ./reports/
```

## Assumptions

- A Python virtualenv already exists on the cluster at `<project_root>/.venv`, with `requirements.txt` installed (`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`), plus `peft` for the LoRA path used by `finetune_encoders.sbatch`.
- `--partition=gpu --gres=gpu:1` matches the target cluster's partition/GRES naming; adjust for your scheduler.
- The full dataset (`data/dataset/train.csv` + `data/Train_Images/`) is already present on cluster-visible storage — `sync_to_cluster.sh` ships a manifest, not the images themselves.
- `src/extract_embeddings.py` and `src/finetune_encoders.py` accept the CLI flags used in the `.sbatch` files below; see [`config/config.yaml`](../config/config.yaml) for the corresponding defaults used by the local CPU run.
