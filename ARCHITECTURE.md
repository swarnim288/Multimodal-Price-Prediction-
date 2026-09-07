# Architecture

Deeper technical notes accompanying the top-level [README](../README.md#architecture). See [PLAN.md](PLAN.md) for the original design rationale and [EXPERIMENTS.md](EXPERIMENTS.md) for per-head results.

## Two-tower rationale

Image and text are structurally different signals with different noise profiles and different costs to model, so they get independent encoders ("towers") rather than a single joint model from the start:

- The **image tower** (ViT) only ever sees pixels; it has no access to the explicit `Value`/`Unit` fields or item name, so anything it contributes is genuinely visual (packaging, product category, apparent quantity/size).
- The **text tower** (MiniLM) only ever sees the catalog text; the explicit quantity/unit fields are additionally parsed out into their own tabular features rather than left for the sentence encoder to infer, since it's cheap and reliable to just parse them directly with a regex instead.

Combining independently-encoded towers via late fusion (rather than an early joint architecture) makes per-modality ablation possible: `hgb_image`, `hgb_text` and `hgb_fused` are the *same* head trained on different feature slices, so the marginal contribution of each modality is a direct, apples-to-apples comparison rather than something that has to be teased out of a single entangled model.

## Why frozen probing before finetuning

Every head in the CPU results table trains on top of **frozen** ViT and MiniLM embeddings — embed once, cache, then train cheap heads on top ("linear probing," extended here to a few shallow non-linear heads: tree ensembles and a small MLP). This ordering is deliberate, not just a limitation of convenience:

1. **It's a real baseline, not a placeholder.** Frozen-encoder probing is a legitimate, widely-used way to measure how much task-relevant structure a general-purpose pretrained encoder already has, before spending compute to finetune it.
2. **It's compute-appropriate.** Every experiment in the results table runs on a CPU laptop against an 8k-row subsample. Backpropagating through a ViT and a transformer text encoder at every training step is not a CPU-laptop-scale operation.
3. **It isolates the head comparison.** Because the representation is fixed, differences between `ridge_text`, `hgb_fused`, `binres_fused` and `mlp_fused` are entirely attributable to the head, not to encoder drift during training.
4. **The finetuning path is still there.** [`slurm/finetune_encoders.sbatch`](../slurm/finetune_encoders.sbatch) scripts LoRA-finetuning both towers jointly with the MLP head on the full dataset — the natural next experiment, just not one that fits the CPU-only constraint this repo's reported numbers are held to.

## Feature block

The fused representation concatenates four pieces. Exact unit-category cardinality and the text-length statistic set are config-driven (see `config/config.yaml`); approximate sizes below:

| Feature block | Approx. dim | Source |
|---|---|---|
| Image embedding | 384 | `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm), frozen, pooled output |
| Text embedding | 384 | `sentence-transformers/all-MiniLM-L6-v2`, frozen, mean-pooled over tokens |
| Quantity value | 1 | regex-parsed `Value:` field, log1p-transformed |
| Unit one-hot | ~15–20 | regex-parsed `Unit:` field, case/alias-normalized, top-N categories + "other" bucket |
| Pack count | 1 | parsed from `(Pack of N)`-style patterns in the item name, defaults to 1 |
| Text-length stats | ~3–5 | character count, word count, bullet-point count of `catalog_content` |

Fused dimensionality is therefore roughly **384 + 384 + ~20–30 ≈ 790–800**, before any head-specific preprocessing (e.g. standardization ahead of Ridge/MLP).

## The bin + residual head, in full

Let `y = log(price)`. On the **training split only**:

1. Fit `K = 10` equal-frequency quantile bin edges `e₀ < e₁ < ... < e_K` over `y`.
2. Assign each training example a bin index `k(y) ∈ {1, ..., K}`.
3. Compute per-bin statistics `μ_k = mean(y | bin = k)` and `σ_k = std(y | bin = k)`.

The head then has two components sharing the fused feature vector `x`:

- a **classifier** producing `p(k | x)`, a softmax posterior over the `K` bins, trained with cross-entropy against the true bin label;
- a **regressor** producing `r̂(x)`, trained to predict the true example's within-bin standardized residual `(y − μ_{k(y)}) / σ_{k(y)}`. Standardizing by each bin's own spread before regression is what makes this *heteroscedastic*: it equalizes target scale across price regimes that would otherwise have very different residual variances (a $2 item and a $2,000 item do not have comparable absolute log-price residuals).

At inference time neither the true bin nor the true residual is known, so the point estimate is decoded as a posterior-weighted expectation in log-space, then mapped back to price:

$$\hat{y} = \exp\left(\sum_{k=1}^{K} p(k \mid x)\cdot \mu_k \;+\; \hat{r}(x)\cdot\bar{\sigma}\right)$$

- `Σₖ p(k|x)·μₖ` is the expected bin location under the model's own uncertainty about which price regime the listing is in.
- `r̂(x)·σ̄` adds back a fine-grained correction, rescaled from standardized-residual space by a fixed, pooled scale `σ̄` (e.g. the mean of `{σ_k}` across bins) — a pooled scale is used because the *true* bin, and hence the true `σ_k`, isn't available at inference.
- The final `exp(...)` inverts the original `log(price)` target transform.

This construction also produces a coarse, free uncertainty signal: the **entropy of `p(k|x)`**. A peaked posterior means the model is confident about the listing's price regime; a flat posterior flags an ambiguous case where the point estimate should be trusted less. This is the natural starting point for the conformal-interval idea listed under [future work](../README.md#explored-directions--future-work).

## Failure modes

- **Image is a weak, indirect price signal.** Product photography in this kind of catalog is largely standardized (studio shots, plain backgrounds); packaging design correlates only loosely with price, and a cheap and an expensive product in the same category can look visually near-identical. Expect `hgb_image` to be the weakest single-modality head, with the image tower mainly contributing pack-format cues (jar vs. bottle vs. multi-pack) rather than absolute price level.
- **Text — especially the parsed quantity fields — carries most of the signal.** Item name and bullet points encode brand and product category; the explicit `Value`/`Unit` fields let a head pick up per-unit pricing patterns directly, without having to infer quantity from prose.
- **Late fusion doesn't guarantee sensible modality weighting.** Concatenating two 384-d embeddings of equal size gives the downstream head no built-in signal about which modality is more informative. `hgb_fused` / `mlp_fused` vs. `hgb_text` is the direct check for whether fusion is actually helping or whether the image block is close to dead weight in the fused vector.
- **The text itself is noisy.** `catalog_content` is free-form seller copy: the [EDA notebook](../notebooks/01_eda.ipynb) finds 99 distinct raw `Unit:` strings — collapsing to 70 after naive lowercasing, still far more than the handful of canonical units (`ounce` / `fl oz` / `count` / `pound` / `gram` / ...) they actually represent — plus a literal `"None"` sentinel string in `Unit` for ~1.25% of rows, and a wide spread in how many bullet points a listing has: at least a quarter have none at all. The regex parser and unit-normalization step have their own noise floor as a result, independent of anything the models do downstream.
- **Heavy tail dominates unprotected metrics.** A small number of very expensive listings (up to $2,796) can dominate MAE and destabilize R² if evaluated on raw price; this is the main reason SMAPE (on price) and RMSLE/R² (on log-price) are used as the headline numbers instead of raw-scale metrics.

## Relationship to the SLURM path

The [feature block](#feature-block) and [bin + residual head](#the-bin--residual-head-in-full) above are architecture, not implementation detail tied to CPU-only execution — the same feature layout and decode formula apply whether the encoders are frozen (as in every reported result here) or finetuned end-to-end (as scripted, not run, in [`slurm/finetune_encoders.sbatch`](../slurm/finetune_encoders.sbatch)). What changes on the GPU path is only where the 384-d embeddings come from, not how they're fused or decoded.
