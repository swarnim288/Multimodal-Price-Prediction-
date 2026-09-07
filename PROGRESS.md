# Progress Log

Dated execution log for this project. See [PLAN.md](PLAN.md) for the design and [EXPERIMENTS.md](EXPERIMENTS.md) for per-head results.

---

## 2026-08-15 — Project revived and completed

This repository consolidates an earlier round of exploratory work — a few finetuning prototypes had been sketched out previously with no completed evaluation (see "Earlier prototypes" in [EXPERIMENTS.md](EXPERIMENTS.md)) — into a documented, reproducible, frozen-encoder pipeline.

Phases below match [PLAN.md §5](PLAN.md#5-execution-timeline):

- [x] **P0** — Data audit, plan, environment setup
- [x] **P1** — Catalog-text parsing, data prep, stratified subsample
- [x] **P2** — Embedding extraction (frozen ViT + MiniLM)
- [x] **P3** — Heads (`ridge_text`, `hgb_image`, `hgb_text`, `hgb_fused`, `binres_fused`, `mlp_fused`) + evaluation + figures
- [x] **P4** — Documentation, EDA notebook, SLURM scale-up scripts, publish
- [ ] **P5** (future, not started) — Full 75k-row run on GPU; LoRA finetune of the ViT tower; DeBERTa-v3 text-tower swap; Qwen2-VL SFT comparison

### P4 detail

- `README.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md` written and cross-linked.
- `notebooks/01_eda.ipynb` executed end-to-end (CSV-only, CPU, no image/embedding loading) against the full training CSV; outputs committed in the notebook itself.
- `slurm/` populated with a sync script and two `sbatch` jobs (embedding extraction, encoder finetuning) for the full-dataset GPU path.
- `data/README.md` + a 200-row `data/sample/train_sample.csv` (seed 42) committed so the repo is inspectable without the full ~75k-row dataset.

## Notes on P5

`slurm/` contains a complete, plausible scale-up path (sync script + two `sbatch` jobs), written to be ready to submit against the target cluster. It has **not been run** — full-dataset and finetuning numbers are not available, and nothing in `README.md` or `EXPERIMENTS.md` claims otherwise. Picking this back up is the natural next step; see [README § Explored directions & future work](../README.md#explored-directions--future-work).
