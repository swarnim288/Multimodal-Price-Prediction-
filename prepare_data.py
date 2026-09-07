"""Stage 1 — load, clean, feature-engineer, and split the raw price dataset.

Pipeline
--------
1. Load the raw CSV (``sample_id, catalog_content, image_link, price``).
2. Parse ``catalog_content`` into item_name / value / unit / details / n_bullets
   (see :mod:`src.parse_text`).
3. Derive the local image filename from ``image_link`` (its last path segment)
   and check it exists under ``images_dir``.
4. Drop rows with a missing image file or non-positive price.
5. Compute ``log_price``, K rank-based quantile bins over ``log_price`` (exactly
   balanced regardless of tied prices), per-bin mu/sigma, and a normalized
   residual — all on the *full* cleaned pool. These ``*_full`` columns are
   descriptive only; ``src/train.py`` recomputes train-split-only bin
   statistics for the actual binres_fused decoding to avoid leaking
   validation targets into the decode constants.
6. Draw a stratified subsample of ``subset_size`` rows (stratified by bin).
7. Stratified train/val split (85/15 by default, stratified by bin).
8. Engineer tabular features: log1p(value), top-K unit one-hot, pack count
   (regex on item name), item-name character length, bullet count.
9. Save ``artifacts/prepared.csv`` (or ``.parquet`` if a parquet engine is
   importable) and ``artifacts/splits.json``.

The saved table is sorted by ``sample_id`` to fix a canonical row order that
``src/extract_embeddings.py`` relies on for positional alignment between the
embedding ``.npy`` files and the prepared table.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from parse_text import extract_pack_count, parse_catalog_content
from utils import load_config, resolve_path, seed_everything

MISSING_UNIT_SLUG = "unk"


def _slugify_unit(unit: str) -> str:
    """Turn a unit string into a safe column-name suffix, e.g. 'Fl Oz' -> 'fl_oz'."""
    u = unit.strip().lower()
    u = re.sub(r"[^a-z0-9]+", "_", u)
    u = u.strip("_")
    return u or MISSING_UNIT_SLUG


def derive_image_filename(image_link: Any) -> str:
    """Last path segment of the image URL/path, e.g. '.../51mo8htwTHL.jpg' -> '51mo8htwTHL.jpg'."""
    if not isinstance(image_link, str) or not image_link:
        return ""
    return image_link.rstrip("/").split("/")[-1]


def build_quantile_bins(log_price: pd.Series, n_bins: int) -> pd.Series:
    """Rank-based quantile binning: always yields exactly `n_bins` near-equal bins.

    Using `.rank(method="first")` before `qcut` avoids the "duplicate bin
    edges" failure mode of `qcut` on data with many tied values (e.g. many
    listings priced at exactly $9.99), since ranks are unique.
    """
    ranks = log_price.rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False)
    return bins.astype(int)


def compute_bin_stats(df: pd.DataFrame, bin_col: str, target_col: str) -> pd.DataFrame:
    """Per-bin mean/std of `target_col`, with a floor on std to avoid div-by-zero."""
    stats = df.groupby(bin_col)[target_col].agg(mu="mean", sigma="std").reset_index()
    stats["sigma"] = stats["sigma"].fillna(0.0).clip(lower=1e-6)
    return stats


def engineer_tabular_features(
    df: pd.DataFrame, top_k_units: int
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add engineered tabular feature columns to `df` (in place-ish, returns a copy).

    Returns (df_with_features, feature_column_names, chosen_unit_labels).
    """
    df = df.copy()

    # value: log1p, missing imputed with the subsample median of non-missing values.
    value_median = df["value"].median(skipna=True)
    if pd.isna(value_median):
        value_median = 0.0
    df["value_filled"] = df["value"].fillna(value_median)
    df["value_log1p"] = np.log1p(df["value_filled"].clip(lower=0))

    # unit: top-K one-hot (case/whitespace-normalized), everything else -> all-zero.
    unit_lower = df["unit"].str.strip().str.lower()
    freq = unit_lower[unit_lower != ""].value_counts()
    top_units = freq.head(top_k_units).index.tolist()
    unit_cols: list[str] = []
    for u in top_units:
        col = f"unit_{_slugify_unit(u)}"
        df[col] = (unit_lower == u).astype(int)
        unit_cols.append(col)

    # pack count: regex on item_name, default 1.
    df["pack_count"] = df["item_name"].apply(extract_pack_count)

    # item name length (characters).
    df["item_name_len"] = df["item_name"].str.len().fillna(0).astype(int)

    # n_bullets already produced by parse_catalog_content.

    feature_cols = ["value_log1p"] + unit_cols + ["pack_count", "item_name_len", "n_bullets"]
    return df, feature_cols, top_units


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full data-prep stage. Returns a summary dict (also printed)."""
    t0 = time.time()
    seed = cfg["seed"]
    seed_everything(seed)

    csv_path = resolve_path(cfg, "csv_path")
    images_dir = resolve_path(cfg, "images_dir")
    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    n_raw = len(df)
    print(f"[prepare_data] raw rows: {n_raw}")

    # --- parse catalog_content -------------------------------------------------
    parsed = df["catalog_content"].apply(parse_catalog_content).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # --- derive image filename + existence check --------------------------------
    df["image_filename"] = df["image_link"].apply(derive_image_filename)
    existing_files = {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()
    df["image_exists"] = df["image_filename"].isin(existing_files)
    n_missing_image = int((~df["image_exists"]).sum())
    print(
        f"[prepare_data] images on disk: {len(existing_files)} | "
        f"rows with a resolvable image file: {int(df['image_exists'].sum())} | "
        f"rows with missing image: {n_missing_image}"
    )

    # --- drop missing image / non-positive price ---------------------------------
    n_bad_price = int((df["price"] <= 0).sum())
    df_clean = df[df["image_exists"] & (df["price"] > 0)].reset_index(drop=True)
    print(
        f"[prepare_data] dropped {n_missing_image} rows (missing image), "
        f"{n_bad_price} rows (price<=0) -> cleaned rows: {len(df_clean)}"
    )

    # --- log price, quantile bins, per-bin stats (full cleaned pool; descriptive) -
    df_clean["log_price"] = np.log(df_clean["price"].astype(float))
    n_bins = cfg["n_bins"]
    df_clean["price_bin"] = build_quantile_bins(df_clean["log_price"], n_bins)
    bin_stats_full = compute_bin_stats(df_clean, "price_bin", "log_price")
    df_clean = df_clean.merge(
        bin_stats_full.rename(columns={"mu": "bin_mu_full", "sigma": "bin_sigma_full"}),
        on="price_bin",
        how="left",
    )
    df_clean["resid_norm_full"] = (
        df_clean["log_price"] - df_clean["bin_mu_full"]
    ) / df_clean["bin_sigma_full"]

    bin_counts_full = df_clean["price_bin"].value_counts().sort_index()
    print(f"[prepare_data] full-pool bin counts:\n{bin_counts_full.to_string()}")

    # --- stratified subsample -----------------------------------------------------
    subset_size = min(cfg["subset_size"], len(df_clean))
    if subset_size < cfg["subset_size"]:
        print(
            f"[prepare_data] WARNING: requested subset_size={cfg['subset_size']} "
            f"exceeds cleaned pool size={len(df_clean)}; using {subset_size}."
        )
    subsample, _ = train_test_split(
        df_clean,
        train_size=subset_size,
        stratify=df_clean["price_bin"],
        random_state=seed,
    )
    subsample = subsample.reset_index(drop=True)
    print(f"[prepare_data] stratified subsample rows: {len(subsample)}")

    # --- stratified train/val split -------------------------------------------------
    val_fraction = cfg["val_fraction"]
    train_df, val_df = train_test_split(
        subsample,
        test_size=val_fraction,
        stratify=subsample["price_bin"],
        random_state=seed,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    combined = pd.concat([train_df, val_df], axis=0)

    # --- engineer tabular features (computed over the full subsample) --------------
    combined, feature_cols, top_units = engineer_tabular_features(combined, cfg["top_k_units"])

    # --- fix canonical row order: sort by sample_id ---------------------------------
    combined = combined.sort_values("sample_id").reset_index(drop=True)

    train_counts = combined.loc[combined["split"] == "train", "price_bin"].value_counts().sort_index()
    val_counts = combined.loc[combined["split"] == "val", "price_bin"].value_counts().sort_index()
    print(f"[prepare_data] train rows: {(combined['split'] == 'train').sum()}, "
          f"val rows: {(combined['split'] == 'val').sum()}")
    print(f"[prepare_data] train bin counts:\n{train_counts.to_string()}")
    print(f"[prepare_data] val bin counts:\n{val_counts.to_string()}")
    print(f"[prepare_data] tabular feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"[prepare_data] top-{cfg['top_k_units']} units: {top_units}")

    # --- persist ---------------------------------------------------------------------
    keep_cols = [
        "sample_id", "item_name", "value", "unit", "details",
        "image_link", "image_filename", "price", "log_price", "price_bin",
        "bin_mu_full", "bin_sigma_full", "resid_norm_full", "split",
    ] + feature_cols  # feature_cols already ends with ..., "n_bullets" -- do not duplicate it above
    out_df = combined[keep_cols]

    prepared_path_parquet = artifacts_dir / "prepared.parquet"
    prepared_path_csv = artifacts_dir / "prepared.csv"
    saved_path: Path
    try:
        import pyarrow  # noqa: F401

        out_df.to_parquet(prepared_path_parquet, index=False)
        saved_path = prepared_path_parquet
    except ImportError:
        out_df.to_csv(prepared_path_csv, index=False)
        saved_path = prepared_path_csv
    print(f"[prepare_data] saved prepared table -> {saved_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    bin_edges_full = {
        int(row.price_bin): {"mu": float(row.mu), "sigma": float(row.sigma)}
        for row in bin_stats_full.itertuples()
    }
    splits_info = {
        "seed": seed,
        "n_raw_rows": n_raw,
        "n_cleaned_rows": len(df_clean),
        "n_missing_image_rows": n_missing_image,
        "n_bad_price_rows": n_bad_price,
        "subset_size_requested": cfg["subset_size"],
        "subset_size_used": subset_size,
        "val_fraction": val_fraction,
        "n_bins": n_bins,
        "n_train": int((combined["split"] == "train").sum()),
        "n_val": int((combined["split"] == "val").sum()),
        "train_sample_ids": combined.loc[combined["split"] == "train", "sample_id"].tolist(),
        "val_sample_ids": combined.loc[combined["split"] == "val", "sample_id"].tolist(),
        "tabular_feature_columns": feature_cols,
        "top_units": top_units,
        "bin_stats_full_pool": bin_edges_full,
        "prepared_table_path": str(saved_path.relative_to(Path(cfg["_repo_root"]))),
        "prepared_table_row_order": "sorted by sample_id ascending",
    }
    splits_path = artifacts_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)
    print(f"[prepare_data] saved split metadata -> {splits_path}")

    elapsed = time.time() - t0
    print(f"[prepare_data] done in {elapsed:.1f}s")
    splits_info["elapsed_sec"] = elapsed
    return splits_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the multimodal price-prediction dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--subset-size", type=int, default=None, help="Override cfg['subset_size']")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subset_size is not None:
        cfg["subset_size"] = args.subset_size
    prepare(cfg)


if __name__ == "__main__":
    main()
