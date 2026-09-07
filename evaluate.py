"""Stage 4 — build evaluation figures from a completed training run.

Reads ``reports/metrics.json`` and ``artifacts/predictions_val.csv`` (both
written by ``src/train.py``) plus the prepared table and image embeddings,
and writes five figures into ``reports/figures/`` using matplotlib only
(dpi=150, tight_layout, no seaborn):

    price_distribution.png  raw price hist (clipped at p99) + log_price hist
    model_comparison.png    horizontal bar of val SMAPE per model, best highlighted
    pred_vs_actual.png      log-log scatter of the best model's val predictions
    smape_by_decile.png     bar of SMAPE per true-price decile, best model
    embedding_pca.png       PCA-2D of val-set image embeddings, colored by log(price)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import load_config, load_prepared_table, resolve_path

COLOR_MAIN = "#4C72B0"
COLOR_ALT = "#DD8452"
COLOR_BEST = "#55A868"


def load_metrics(reports_dir: Path) -> dict[str, Any]:
    with open(reports_dir / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(artifacts_dir: Path) -> pd.DataFrame:
    return pd.read_csv(artifacts_dir / "predictions_val.csv")


def plot_price_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    p99 = df["price"].quantile(0.99)
    clipped = df["price"].clip(upper=p99)
    axes[0].hist(clipped, bins=50, color=COLOR_MAIN, edgecolor="white", linewidth=0.5)
    axes[0].set_title(f"Raw price distribution (clipped at p99 = {p99:.0f})")
    axes[0].set_xlabel("price")
    axes[0].set_ylabel("count")

    axes[1].hist(df["log_price"], bins=50, color=COLOR_ALT, edgecolor="white", linewidth=0.5)
    axes[1].set_title("log(price) distribution")
    axes[1].set_xlabel("log(price)")
    axes[1].set_ylabel("count")

    fig.suptitle(f"Price distribution (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics: dict[str, Any], out_path: Path) -> None:
    models = metrics["models"]
    names = list(models.keys())
    smapes = [models[n]["smape"] for n in names]
    best = metrics["best_model"]

    # Sort so the best (lowest SMAPE) model ends up at the TOP of the barh chart:
    # argsort gives ascending order (best first); reversing makes the best the
    # LAST element, and matplotlib's barh draws the last list element at the top.
    order = np.argsort(smapes)[::-1]
    names_sorted = [names[i] for i in order]
    smapes_sorted = [smapes[i] for i in order]
    colors = [COLOR_BEST if n == best else COLOR_MAIN for n in names_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names_sorted, smapes_sorted, color=colors)
    ax.set_xlabel("Validation SMAPE (%)")
    ax.set_title("Model comparison -- validation SMAPE (lower is better)")
    for bar, val in zip(bars, smapes_sorted):
        ax.text(bar.get_width() + max(smapes_sorted) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, max(smapes_sorted) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(pred_df: pd.DataFrame, best_model: str, out_path: Path) -> None:
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df[f"y_pred_{best_model}"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color=COLOR_MAIN, edgecolor="none")
    lo = max(min(float(y_true.min()), float(y_pred.min())), 1e-2)
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual price (log scale)")
    ax.set_ylabel("Predicted price (log scale)")
    ax.set_title(f"Predicted vs. actual price -- {best_model} (val set, n={len(y_true)})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_smape_by_decile(metrics: dict[str, Any], out_path: Path) -> None:
    per_decile = metrics["per_decile"]
    smapes = [d["smape"] for d in per_decile]
    labels = [f"D{d['decile']}\n${d['price_min']:.1f}-{d['price_max']:.0f}" for d in per_decile]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, smapes, color=COLOR_MAIN)
    ax.set_xlabel("True-price decile (low -> high price)")
    ax.set_ylabel("SMAPE (%)")
    ax.set_title(f"SMAPE by true-price decile -- {metrics['best_model']} (val set)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_pca(df: pd.DataFrame, emb_image: np.ndarray, out_path: Path) -> None:
    val_mask = (df["split"] == "val").to_numpy()
    X_val = emb_image[val_mask]
    log_price_val = df.loc[val_mask, "log_price"].to_numpy()

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_val)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=log_price_val, cmap="viridis", s=8, alpha=0.7, edgecolor="none"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(price)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(f"PCA of image embeddings (val set, n={len(X_val)}), colored by log(price)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluation(cfg: dict[str, Any]) -> None:
    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(reports_dir)
    pred_df = load_predictions(artifacts_dir)
    df = load_prepared_table(cfg)
    emb_image = np.load(artifacts_dir / "emb_image.npy")
    if len(emb_image) != len(df):
        raise ValueError(f"emb_image has {len(emb_image)} rows but prepared table has {len(df)}.")

    best_model = metrics["best_model"]
    print(f"[evaluate] best_model={best_model}")

    plot_price_distribution(df, figures_dir / "price_distribution.png")
    print("[evaluate] wrote price_distribution.png")

    plot_model_comparison(metrics, figures_dir / "model_comparison.png")
    print("[evaluate] wrote model_comparison.png")

    plot_pred_vs_actual(pred_df, best_model, figures_dir / "pred_vs_actual.png")
    print("[evaluate] wrote pred_vs_actual.png")

    plot_smape_by_decile(metrics, figures_dir / "smape_by_decile.png")
    print("[evaluate] wrote smape_by_decile.png")

    plot_embedding_pca(df, emb_image, figures_dir / "embedding_pca.png")
    print("[evaluate] wrote embedding_pca.png")

    print(f"[evaluate] all figures written to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures from a completed training run.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
