# Run Notes

Factual log of the pipeline run behind every number in this repository.
All figures below are copied from actual command output or from
`reports/metrics.json` / `artifacts/splits.json`, both written by the code in
`src/`. Run date: 2026-08-15.

Environment: local CPU-only machine — torch 2.8.0+cpu, timm 1.0.20,
transformers 4.57.0, torchvision 0.23.0+cpu, scikit-learn 1.9.0, pandas
2.3.3, matplotlib 3.11.1, 12 logical CPUs (`torch.set_num_threads(12)`).

## 1. Config used

Full file: `config/config.yaml`.

| Knob | Value |
|---|---|
| seed | 42 |
| subset_size | 8000 |
| n_bins | 10 |
| val_fraction | 0.15 |
| top_k_units | 8 |
| image_model | `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm, pretrained, num_classes=0, 384-d) |
| text_model | `sentence-transformers/all-MiniLM-L6-v2` (mean-pooled, 384-d) |
| image_batch_size / text_batch_size | 32 / 64 |
| text_max_length | 256 |
| mlp | 512→128→1, GELU, dropout 0.2, AdamW lr 1e-3 wd 1e-4, batch 256, max 60 epochs, patience 8 |

## 2. Data preparation (`src/prepare_data.py`, 22.4s)

- Raw CSV: 75,000 rows.
- **Image availability, verified per row** (`os.listdir` on `Train_Images/` +
  set-membership against every row's derived filename): 75,000 rows resolve to
  72,288 unique image filenames — 2,712 rows share an image file with another
  row (duplicate product photos across listing variants) — of which 72,287
  exist on disk. Row-level result: **74,999 of 75,000 rows have a resolvable,
  existing image file; exactly 1 row does not** (`51mjZYDYjyL.jpg`, missing
  from disk). That row was dropped.
- Rows with `price <= 0`: 0.
- Cleaned pool: **74,999 rows**.
- `log_price = log(price)`; K=10 rank-based quantile bins (exactly balanced:
  7500/7500/.../7499/7500 across bins 0-9) computed on the full cleaned pool;
  per-bin mu/sigma and a normalized residual stored as `*_full` columns
  (descriptive only — see §4 for why the model itself does not use these).
- Stratified subsample: **8000 rows** (stratified by bin, seed 42).
- Stratified 85/15 train/val split (by bin): **6800 train / 1200 val**, exactly
  680/120 per bin in every one of the 10 bins.
- Tabular features engineered (12 columns): `value_log1p` (median-imputed
  before log1p, 97/8000 rows had no parseable `Value:`), 8 top-unit one-hot
  columns (`unit_ounce, unit_count, unit_fl_oz, unit_oz, unit_pound, unit_ct,
  unit_lb, unit_sq_ft`), `pack_count` (regex `Pack of (\d+)` on item name,
  default 1), `item_name_len`, `n_bullets`.
- Output: `artifacts/prepared.csv` (CSV rather than parquet — no parquet
  engine in the environment; 8000 rows x 26 cols) + `artifacts/splits.json`.
- One bug caught and fixed before any downstream stage ran: an early version
  of the column list included `n_bullets` twice, which pandas silently
  suffixed to `n_bullets.1` on CSV round-trip. Fixed in `prepare_data.py`
  before any embeddings/training used the table (all numbers in this document
  come from the corrected, re-run version).

## 3. Embedding extraction (`src/extract_embeddings.py`)

### Text (MiniLM), single call, all 8000 rows
- Model load: 7.6s. Processing: **339.4s** (8000 rows → 23.57 texts/sec).
- Output: `artifacts/emb_text.npy`, shape (8000, 384), float32, no NaNs.

### Image (ViT), benchmarked then chunked
- **Pre-flight benchmark (64 images):** 9.87s → **6.48 imgs/sec** → projected
  **20.6 minutes** for the full 8000, acceptable for a CPU run, so
  `subset_size=8000` and the ViT-S/16 model were used as configured, with no
  fallback to a lighter backbone.
- Extraction was chunked into 4 x 2000-row calls (keeps each command
  restartable and bounds the cost of any mid-run failure), then merged with
  `--merge`:

  | Chunk | Rows | Wall time | Throughput | Corrupt/missing |
  |---|---|---|---|---|
  | 1 | [0, 2000) | 266.7s | 7.50 img/s | 1 (`sample_id=31048`, `71UswDUCTuL.jpg`) |
  | 2 | [2000, 4000) | 269.9s | 7.41 img/s | 1 (`sample_id=78307`, `81o1nWoPkBL.jpg`) |
  | 3 | [4000, 6000) | 267.9s | 7.47 img/s | 1 (`sample_id=156035`, `81QAU9pGoVL.jpg`) |
  | 4 | [6000, 8000) | 269.6s | 7.42 img/s | 0 |

  Total processing time across chunks: **1074.1s (~17.9 min)**. Achieved
  throughput (~7.4-7.5 img/s) was slightly higher than the 64-image benchmark
  (6.48 img/s), consistent with fixed per-call overhead dominating such a
  small benchmark sample.
- **3 of 8000 images (0.0375%) were present on disk but failed to decode**
  (PIL exceptions) — corrupt files, distinct from the single genuinely-missing
  file filtered out in `prepare_data.py`. Each was replaced with an exact
  all-zero 384-d vector and logged (`artifacts/image_extract_errors.log`).
  Verified post-merge: all 3 rows are exactly all-zero at their correct global
  positions and every other row is non-zero, with no NaNs anywhere in the
  merged array.
- Merge: `artifacts/emb_image.npy`, shape (8000, 384), float32, verified
  contiguous [0, 8000) coverage from the 4 chunk files. No chunk was re-run;
  no data was re-extracted.

## 4. Training (`src/train.py`, 185.6s total)

All 6 heads trained on the 6800-row train split, evaluated once on the
1200-row val split. `binres_fused`'s bin mu_k/sigma_k were recomputed from
scratch on **train-split rows only** (not the `*_full` columns from
`prepare_data.py`, which are computed on the full pre-split pool and would
leak validation targets into the decode constants).

| Model | SMAPE (%) | MAE | RMSLE | R2 (log) | Fit time |
|---|---|---|---|---|---|
| ridge_text | 66.07 | 14.94 | 0.861 | 0.158 | 0.04s |
| hgb_image | 64.24 | 14.23 | 0.834 | 0.209 | 11.4s |
| hgb_text | 63.32 | 14.09 | 0.826 | 0.223 | 10.0s |
| hgb_fused | 59.53 | 13.16 | 0.772 | 0.319 | 11.8s |
| **binres_fused (best)** | **57.83** | **12.85** | **0.755** | **0.349** | 141.1s |
| mlp_fused | 59.94 | 13.84 | 0.796 | 0.287 | 7.8s |

`mlp_fused` early-stopped at epoch 12 (patience 8 on val SMAPE), best weights
from epoch 4 (val SMAPE 59.94) restored before scoring — see
`artifacts/models/mlp_fused_history.json` for the full per-epoch log.

Sanity checks: text beats image (63.32 vs 64.24); every fused model beats
every single-modality model; the distribution-aware head beats direct fused
regression. No debugging of the modeling stage was required.

### Per-decile SMAPE, best model (`binres_fused`)

| Decile | Price range | n | SMAPE (%) |
|---|---|---|---|
| D0 | $0.50 - 3.52 | 120 | 97.38 |
| D1 | $3.58 - 5.60 | 120 | 71.93 |
| D2 | $5.65 - 7.99 | 120 | 50.28 |
| D3 | $7.99 - 10.75 | 120 | 36.84 |
| D4 | $10.85 - 13.99 | 120 | 29.44 |
| D5 | $14.09 - 18.00 | 120 | 29.14 |
| D6 | $18.47 - 24.64 | 120 | 41.44 |
| D7 | $24.69 - 33.85 | 120 | 57.48 |
| D8 | $33.99 - 52.30 | 120 | 70.71 |
| D9 | $52.41 - 298.00 | 120 | 93.70 |

Classic U-shape: the model is most accurate in the mid-price deciles (D4-D5,
~29% SMAPE) and least accurate at both price extremes — cheapest items (SMAPE
inflates because small absolute errors are large relative errors) and the
long, sparse expensive tail (D9 spans $52-298 in only 120 points).

## 5. Evaluation figures (`src/evaluate.py`)

All 5 figures written to `reports/figures/` (matplotlib, dpi=150,
tight_layout) and visually spot-checked:

- `price_distribution.png` — raw price (clipped at p99=$145) + log_price, n=8000.
- `model_comparison.png` — horizontal SMAPE bar chart, `binres_fused` highlighted.
- `pred_vs_actual.png` — log-log scatter for `binres_fused`, alpha=0.3, y=x line.
- `smape_by_decile.png` — bar chart matching the table above.
- `embedding_pca.png` — 2D PCA of val-set image embeddings (PC1 6.2% var, PC2
  4.9% var) colored by log(price); no dramatic price-separated clustering,
  which is expected for only 2 of 384 dimensions and a frozen,
  ImageNet-pretrained (not price-finetuned) backbone.

## 6. Wall-clock summary

| Stage | Time |
|---|---|
| prepare_data.py | 22.4s |
| extract_embeddings.py --modality text | 339.4s (+7.6s model load) |
| extract_embeddings.py --modality image (benchmark) | 9.87s |
| extract_embeddings.py --modality image (4 chunks + merge) | 1074.1s processing (+~16s model load/chunk) |
| train.py (all 6 heads) | 185.6s |
| evaluate.py (5 figures) | under a minute |
| **Total measured compute** | **~28 minutes** across all stages |

(Not included above: two one-time model downloads from the HF Hub during
initial smoke-testing of the extraction code on 8-16 row samples, ~24-31s
each, before any of the timed runs in this document.)
