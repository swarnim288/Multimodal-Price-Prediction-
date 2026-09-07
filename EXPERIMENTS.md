# Experiments

Per-head experiment log. All runs share the same stratified 8,000-sample subsample, 85/15 split (6,800 train / 1,200 val), and seeds defined in `config/config.yaml`; see [ARCHITECTURE.md](ARCHITECTURE.md) for head details and [PLAN.md](PLAN.md) for the overall design.

> **All metric cells below are copied verbatim from [`reports/metrics.json`](../reports/metrics.json), produced by the run of 2026-08-15 (see [`reports/RUN_NOTES.md`](../reports/RUN_NOTES.md) for the full run log). The same table is mirrored in [README.md § Results](../README.md#results).**

---

## `ridge_text`

- **Setup:** Ridge regression (L2-regularized linear regression) on the MiniLM text embedding only (384-d), predicting log-price. Engineered tabular features enter only the fused heads.
- **Hypothesis:** A linear baseline on text alone should already capture a meaningful chunk of the signal, since brand/category/quantity cues are largely legible from text. Sets the floor the non-linear and fused heads need to beat.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 66.07 | 14.94 | 0.861 | 0.158 |

- **Notes:** Weakest head overall, as expected for a linear probe on a frozen sentence embedding — `hgb_text` takes a further 2.7 SMAPE points off the identical features, so the representation carries non-linear structure a Ridge probe can't express.

## `hgb_image`

- **Setup:** HistGradientBoosting regressor on the image feature block only (frozen ViT embedding, no tabular/text features), predicting log-price.
- **Hypothesis:** Expected to be the weakest head in the comparison — see [failure modes](ARCHITECTURE.md#failure-modes). Should beat a constant-prediction baseline but trail every text-informed head by a wide margin.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 64.24 | 14.23 | 0.834 | 0.209 |

- **Notes:** The hypothesis was half wrong, in an interesting way: the image-only head trails every *non-linear* text head as predicted, but actually **edges out the linear text baseline** (64.24 vs 66.07 SMAPE) — a frozen, ImageNet-pretrained ViT read through a tree ensemble carries more price signal than a linear probe of the text embedding. Packaging apparently isn't as uninformative as assumed.

## `hgb_text`

- **Setup:** HistGradientBoosting regressor on the MiniLM text embedding only (same features as `ridge_text`), predicting log-price.
- **Hypothesis:** A non-linear model on the same features as `ridge_text` should pick up interaction effects (e.g. unit × value) a linear model can't; expected to beat `ridge_text`.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 63.32 | 14.09 | 0.826 | 0.223 |

- **Notes:** Confirmed — beats `ridge_text` by 2.7 SMAPE points on identical features, and is the strongest single-modality head.

## `hgb_fused`

- **Setup:** HistGradientBoosting regressor on the full fused feature vector (image + text + tabular), predicting log-price.
- **Hypothesis:** Should match or beat `hgb_text`; the size of the gap (or lack of one) between the two is the direct measurement of how much the image tower contributes once text is already available.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 59.53 | 13.16 | 0.772 | 0.319 |

- **Notes:** Fusion is clearly worth it: −3.8 SMAPE vs `hgb_text` and −4.7 vs `hgb_image`. The image tower and the engineered quantity/unit features contribute complementary signal on top of the text embedding rather than being redundant with it.

## `binres_fused`

- **Setup:** Distribution-aware bin-classifier + within-bin residual regressor on the full fused feature vector; decoded via the posterior-weighted expectation `ŷ = exp(Σₖ p(k|x)·μₖ + r̂·σ̄)` (see [ARCHITECTURE.md](ARCHITECTURE.md#the-bin--residual-head-in-full)).
- **Hypothesis:** The core research question of this repo — does discretizing the regression into a coarse classification plus a fine residual outperform regressing log-price directly with a comparably-sized model? Also expected to yield a usable bin-posterior-entropy uncertainty signal even if point-estimate accuracy roughly ties `hgb_fused`.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 57.83 | 12.85 | 0.755 | 0.349 |

- **Notes:** **Best head overall** — another −1.7 SMAPE on top of `hgb_fused`, so the discretize-then-refine structure does help beyond what the same features + a direct regressor achieve. Costs the most compute of any head (141s of the 186s training stage, since it fits a 10-class classifier plus a residual regressor). Per-decile SMAPE is strongly U-shaped (≈29% in the mid-price deciles vs ≈93–97% at the two extremes — see `reports/figures/smape_by_decile.png`), which is the expected signature of a heavy-tailed target: relative error explodes for very cheap items and the sparse expensive tail. Bin-posterior entropy as an uncertainty signal is logged but not yet analyzed against error — first item of future work.

## `mlp_fused`

- **Setup:** Small feed-forward network (2–3 hidden layers) on the full fused feature vector, predicting log-price directly.
- **Hypothesis:** A neural head has capacity for feature interactions the tree ensemble might miss — or might not, since HistGB is already a strong tabular baseline. Primarily a check on whether `hgb_fused` is close to the ceiling for this feature set.
- **Result:**

  | SMAPE | MAE | RMSLE | R² |
  |---|---|---|---|
  | 59.94 | 13.84 | 0.796 | 0.287 |

- **Notes:** Early-stopped at epoch 12 with best weights from epoch 4 (patience 8 on val SMAPE; full per-epoch log in `artifacts/models/mlp_fused_history.json`). Lands between `hgb_text` and `hgb_fused` — consistent with the hypothesis that HistGB is already near the ceiling for this fixed frozen-feature set, and the marginal gains lie in *better features* (finetuned encoders) rather than a bigger head.

---

## Earlier prototypes (not completed — no numbers)

These were explored earlier in the project's history; configs and/or partial weights exist locally, but none reached a completed, evaluated run against the SMAPE/MAE/RMSLE/R² suite above. No results are reported for any of them, here or anywhere else in this repo.

### Qwen2-VL-2B-Instruct, LoRA-finetuned

Framed price prediction as generation rather than regression: image + catalog text in, a price string out, fit with a LoRA adapter over Qwen2-VL-2B-Instruct. Prototyped (adapter config and partial weights exist locally); not evaluated. Revisiting this would need a decoding/parsing strategy for turning generated text back into a numeric price, plus a fair way to compare a generative model's error distribution against the regression heads above.

### DeBERTa-v3 text tower

Explored as a drop-in replacement for MiniLM ahead of fusion, on the hypothesis that a larger, more expressive text encoder would move `hgb_text` / `hgb_fused`. Prototyped; not benchmarked against the MiniLM-based heads above.

### End-to-end ViT-B/16 finetuning

Explored finetuning the image tower directly (rather than frozen ViT-S/16 probing) to measure how much frozen probing leaves on the table specifically for the image modality. Prototyped; not benchmarked. `slurm/finetune_encoders.sbatch` scripts a version of this jointly with the text tower and fusion head, on the full dataset — also not run.

---

## Provenance of these numbers

Every metric above — and the mirrored table in [README.md](../README.md#results) — is copied verbatim from `reports/metrics.json`, written by `src/evaluate.py` at the end of the run logged in [`reports/RUN_NOTES.md`](../reports/RUN_NOTES.md). No rounding or selection beyond what `evaluate.py` itself reports.
