# Project Plan — Multimodal Price Prediction

> Working plan for building, evaluating, and documenting the multimodal price
> regression pipeline. Kept up to date as the project evolves; see
> [PROGRESS.md](PROGRESS.md) for the execution log.

## 1. Problem statement

Given a product listing consisting of:

- a **catalog text** blob (item name, marketing bullet points, quantity value + unit),
- a **product image**,

predict the listing **price** (positive, heavy-tailed, in currency units).

Primary metric: **SMAPE** (Symmetric Mean Absolute Percentage Error), plus MAE,
RMSLE and R² on log-price as secondary diagnostics. SMAPE is the natural choice
for heavy-tailed price targets because it is scale-free and bounded (0–200).

## 2. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target transform | `log(price)` | Price is log-normal-ish, heavy right tail (median ≈ 14, max ≈ 2800) |
| Encoders | Frozen pretrained ViT (image) + MiniLM sentence encoder (text) | Frozen-backbone probing is compute-efficient and a strong baseline; full finetuning reserved for the GPU path |
| Fusion | Late fusion (concatenation of modality embeddings + engineered tabular features) | Simple, interpretable, lets us ablate each modality's contribution |
| Heads | (a) Ridge text-only baseline, (b) gradient-boosted trees on fused features, (c) distribution-aware quantile-bin classifier + within-bin residual regressor, (d) MLP fusion head | Covers a linear baseline → tree ensemble → discretized-regression research idea → neural head |
| Experiment scale | Stratified subsample (≈ 8–10k of 75k) on CPU | Full-scale runs are scripted for SLURM (see `slurm/`), but every reported number must come from a run that actually finished |

### The distribution-aware head (core idea carried over from early experiments)

Instead of regressing log-price directly, discretize it into **K = 10 quantile
bins**, then:

1. classify the bin (posterior `p(k|x)` over bins),
2. regress the **within-bin normalized residual** `(log p − μ_k)/σ_k`,
3. decode with the posterior-weighted expectation
   `ŷ = exp( Σ_k p(k|x)·μ_k + r̂·σ̄ )`.

This is a form of **discretized ordinal regression with heteroscedastic residual
normalization** — it stabilizes the target, gives the model an easier
classification sub-problem, and produces a coarse uncertainty signal (bin
posterior entropy) for free.

## 3. Pipeline stages

```
raw CSV (75k rows)
  → parse catalog_content (item name / bullets / value / unit)
  → clean + log-price + quantile bins
  → stratified subsample (n=8000) + train/val split (85/15, stratified by bin)
  → image embeddings   (frozen ViT,   timm)      ─┐
  → text embeddings    (MiniLM mean-pooled)       ├─ late fusion
  → tabular features   (value, unit, pack count)  ─┘
  → heads: ridge / HistGB / bin+residual / MLP
  → evaluation (SMAPE, MAE, RMSLE, R²) + figures
```

## 4. Repository layout

```
src/                 modular pipeline (one stage per file, config-driven)
config/config.yaml   all knobs in one place
scripts/             one-command local run
slurm/               full-dataset GPU path (sbatch + sync scripts)
notebooks/           EDA
reports/             metrics.json + figures produced by the last run
docs/                this plan, experiment log, architecture notes
data/                schema docs + tiny sample (full data not committed)
```

## 5. Execution timeline

| Phase | Content | Status |
|---|---|---|
| P0 | Data audit, plan, environment | done |
| P1 | Text parsing + data prep + subsample | done |
| P2 | Embedding extraction (ViT + MiniLM) | done |
| P3 | Heads + evaluation + figures | done |
| P4 | Docs, EDA notebook, SLURM scripts, publish | done |
| P5 (future) | Full 75k run on GPU; LoRA finetune of ViT; DeBERTa-v3 text tower; Qwen2-VL SFT comparison | scripted, not run |

## 6. Constraints & principles

- **CPU-only local machine** — every default in `config.yaml` must finish locally.
- **No fabricated numbers** — `reports/metrics.json` is written by `evaluate.py`
  at the end of a real run; the README results table mirrors it verbatim.
- **Reproducibility** — fixed seeds, config-driven, deterministic splits.
