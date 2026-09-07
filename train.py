"""Stage 3 — train every model head, evaluate on the held-out val split.

Heads (fused = concat[image_emb, text_emb, tabular]; each feature block is
standardized with a StandardScaler fit on the train split only):

    a. ridge_text    - Ridge(alpha=1.0) on text embeddings -> log_price
    b. hgb_image     - HistGradientBoostingRegressor on image embeddings
    c. hgb_text      - HistGradientBoostingRegressor on text embeddings
    d. hgb_fused     - HistGradientBoostingRegressor on fused features
    e. binres_fused  - distribution-aware head: HistGradientBoostingClassifier
                        over the K quantile bins (fused features) +
                        HistGradientBoostingRegressor on the normalized
                        within-bin residual (fused features); decoded as
                            y_hat = exp( sum_k p(k|x) mu_k + r_hat * sum_k p(k|x) sigma_k )
                        with mu_k / sigma_k computed on the TRAIN split only
                        (the prepare_data.py `bin_mu_full`/`bin_sigma_full`
                        columns are full-pool statistics and are NOT used
                        here, to avoid leaking validation targets into the
                        decode constants).
    f. mlp_fused     - torch MLP (in -> 512 -> 128 -> 1, GELU, dropout 0.2),
                        AdamW, batch 256, up to 60 epochs, early stopping
                        (patience 8) on val SMAPE.

All heads predict log_price; predictions are back-transformed with exp().
Writes reports/metrics.json, artifacts/predictions_val.csv, and saves every
fitted model/scaler under artifacts/models/.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils import load_config, load_prepared_table, resolve_path, rmsle, seed_everything, smape

LOG_CLIP = 10.0  # defensive clip on predicted log_price before exp(), avoids inf/NaN from outlier predictions


def to_price(y_log: np.ndarray) -> np.ndarray:
    """exp() back-transform with a defensive clip so pathological predictions can't overflow."""
    return np.exp(np.clip(y_log, -LOG_CLIP, LOG_CLIP))


def compute_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    return {
        "smape": smape(y_true_price, y_pred_price),
        "mae": float(mean_absolute_error(y_true_price, y_pred_price)),
        "rmsle": rmsle(y_true_price, y_pred_price),
        "r2": float(r2_score(y_true_log, y_pred_log)),
    }


def build_features(df: pd.DataFrame, emb_image: np.ndarray, emb_text: np.ndarray, tabular_cols: list[str]):
    X_image = emb_image.astype(np.float32)
    X_text = emb_text.astype(np.float32)
    X_tab = df[tabular_cols].to_numpy(dtype=np.float32)
    X_fused = np.concatenate([X_image, X_text, X_tab], axis=1)
    return X_image, X_text, X_tab, X_fused


def fit_scaler(train_arr: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_arr)
    return scaler


# ---------------------------------------------------------------------------
# binres_fused: distribution-aware quantile-bin + residual head
# ---------------------------------------------------------------------------

def compute_train_bin_stats(train_log_price: np.ndarray, train_bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean/std of log_price, computed on TRAIN rows only.

    Returns (mu_arr, sigma_arr), each shape (n_bins,), indexed by bin id.
    Falls back to the global train mean/std for any bin with < 2 train
    samples (should not happen with a stratified split, but guarded).
    """
    global_mu = float(np.mean(train_log_price))
    global_sigma = float(np.std(train_log_price, ddof=1)) if len(train_log_price) > 1 else 1.0
    global_sigma = max(global_sigma, 1e-6)

    mu_arr = np.full(n_bins, global_mu, dtype=np.float64)
    sigma_arr = np.full(n_bins, global_sigma, dtype=np.float64)
    for k in range(n_bins):
        mask = train_bins == k
        n_k = int(mask.sum())
        if n_k >= 2:
            mu_arr[k] = float(np.mean(train_log_price[mask]))
            sigma_arr[k] = max(float(np.std(train_log_price[mask], ddof=1)), 1e-6)
        elif n_k == 1:
            mu_arr[k] = float(train_log_price[mask][0])
            sigma_arr[k] = global_sigma
            print(f"[binres_fused] WARNING: bin {k} has only 1 train sample; using global sigma fallback.")
        else:
            print(f"[binres_fused] WARNING: bin {k} has 0 train samples; using global mu/sigma fallback.")
    return mu_arr, sigma_arr


def train_binres_fused(
    X_fused_train: np.ndarray,
    X_fused_val: np.ndarray,
    train_bins: np.ndarray,
    y_train_log: np.ndarray,
    n_bins: int,
    seed: int,
):
    mu_arr, sigma_arr = compute_train_bin_stats(y_train_log, train_bins, n_bins)

    resid_train = (y_train_log - mu_arr[train_bins]) / sigma_arr[train_bins]

    classifier = HistGradientBoostingClassifier(random_state=seed)
    classifier.fit(X_fused_train, train_bins)

    regressor = HistGradientBoostingRegressor(random_state=seed)
    regressor.fit(X_fused_train, resid_train)

    proba_val = classifier.predict_proba(X_fused_val)  # columns follow classifier.classes_ order
    class_order = classifier.classes_.astype(int)
    mu_for_classes = mu_arr[class_order]
    sigma_for_classes = sigma_arr[class_order]

    weighted_mu = proba_val @ mu_for_classes
    weighted_sigma = proba_val @ sigma_for_classes
    r_hat_val = regressor.predict(X_fused_val)

    y_pred_log_val = weighted_mu + r_hat_val * weighted_sigma

    extras = {
        "mu_k": mu_arr.tolist(),
        "sigma_k": sigma_arr.tolist(),
        "classes_": class_order.tolist(),
    }
    return classifier, regressor, y_pred_log_val, extras


# ---------------------------------------------------------------------------
# mlp_fused: torch MLP head
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int, hidden1: int, hidden2: int, dropout: float):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


def train_mlp_fused(
    mlp_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    y_val_price: np.ndarray,
    seed: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    model = _build_mlp(X_train.shape[1], mlp_cfg["hidden1"], mlp_cfg["hidden2"], mlp_cfg["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_cfg["lr"], weight_decay=mlp_cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=mlp_cfg["batch_size"], shuffle=True
    )

    best_smape = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float]] = []
    epoch = -1

    for epoch in range(mlp_cfg["max_epochs"]):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]
        train_mse = running_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred_log = model(X_val_t).squeeze(-1).numpy()
        val_pred_price = to_price(val_pred_log)
        val_smape = smape(y_val_price, val_pred_price)
        history.append({"epoch": epoch + 1, "train_mse": train_mse, "val_smape": val_smape})

        improved = val_smape < best_smape - 1e-6
        if improved:
            best_smape, best_epoch = val_smape, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[mlp_fused] epoch {epoch + 1}/{mlp_cfg['max_epochs']} "
            f"train_mse={train_mse:.4f} val_smape={val_smape:.2f} "
            f"best_smape={best_smape:.2f} patience={patience_counter}/{mlp_cfg['patience']}"
        )
        if patience_counter >= mlp_cfg["patience"]:
            print(f"[mlp_fused] early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_log_val = model(X_val_t).squeeze(-1).numpy()

    meta = {"best_epoch": best_epoch + 1, "epochs_trained": epoch + 1, "history": history}
    return model, y_pred_log_val, meta


# ---------------------------------------------------------------------------
# Per-decile SMAPE for the best model
# ---------------------------------------------------------------------------

def per_decile_smape(y_true_price: np.ndarray, y_pred_price: np.ndarray, n_deciles: int = 10) -> list[dict[str, Any]]:
    ranks = pd.Series(y_true_price).rank(method="first")
    decile = pd.qcut(ranks, q=n_deciles, labels=False).to_numpy()
    out = []
    for d in range(n_deciles):
        mask = decile == d
        out.append(
            {
                "decile": int(d),
                "n": int(mask.sum()),
                "price_min": float(y_true_price[mask].min()),
                "price_max": float(y_true_price[mask].max()),
                "smape": smape(y_true_price[mask], y_pred_price[mask]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    t_start = time.time()
    seed = cfg["seed"]
    seed_everything(seed)

    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_prepared_table(cfg)
    with open(artifacts_dir / "splits.json", "r", encoding="utf-8") as f:
        splits_info = json.load(f)
    tabular_cols = splits_info["tabular_feature_columns"]
    n_bins = splits_info["n_bins"]

    emb_image_path = artifacts_dir / "emb_image.npy"
    emb_text_path = artifacts_dir / "emb_text.npy"
    if not emb_image_path.exists() or not emb_text_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings. Expected {emb_image_path} and {emb_text_path}. "
            "Run src/extract_embeddings.py (and --merge if chunked) first."
        )
    emb_image = np.load(emb_image_path)
    emb_text = np.load(emb_text_path)
    if len(emb_image) != len(df) or len(emb_text) != len(df):
        raise ValueError(
            f"Row mismatch: prepared table has {len(df)} rows, "
            f"emb_image has {len(emb_image)}, emb_text has {len(emb_text)}."
        )
    print(f"[train] prepared rows={len(df)}, image_emb_dim={emb_image.shape[1]}, text_emb_dim={emb_text.shape[1]}, "
          f"tabular_dim={len(tabular_cols)}")

    X_image, X_text, X_tab, X_fused = build_features(df, emb_image, emb_text, tabular_cols)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    print(f"[train] n_train={train_mask.sum()}, n_val={val_mask.sum()}")

    y_log = df["log_price"].to_numpy(dtype=np.float64)
    y_price = df["price"].to_numpy(dtype=np.float64)
    y_train_log, y_val_log = y_log[train_mask], y_log[val_mask]
    y_train_price, y_val_price = y_price[train_mask], y_price[val_mask]
    train_bins = df["price_bin"].to_numpy(dtype=int)[train_mask]

    val_sample_ids = df.loc[val_mask, "sample_id"].to_numpy()

    # --- scalers, fit on TRAIN only ------------------------------------------------
    scaler_text = fit_scaler(X_text[train_mask])
    scaler_image = fit_scaler(X_image[train_mask])
    scaler_fused = fit_scaler(X_fused[train_mask])

    X_text_train, X_text_val = scaler_text.transform(X_text[train_mask]), scaler_text.transform(X_text[val_mask])
    X_image_train, X_image_val = scaler_image.transform(X_image[train_mask]), scaler_image.transform(X_image[val_mask])
    X_fused_train, X_fused_val = scaler_fused.transform(X_fused[train_mask]), scaler_fused.transform(X_fused[val_mask])

    results: dict[str, dict[str, Any]] = {}
    val_preds_price: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    # --- a. ridge_text ---------------------------------------------------------------
    t0 = time.time()
    ridge_text = Ridge(alpha=1.0, random_state=seed)
    ridge_text.fit(X_text_train, y_train_log)
    pred_log = ridge_text.predict(X_text_val)
    pred_price = to_price(pred_log)
    results["ridge_text"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    val_preds_price["ridge_text"] = pred_price
    timings["ridge_text"] = time.time() - t0
    joblib.dump(ridge_text, models_dir / "ridge_text.joblib")
    print(f"[train] ridge_text done in {timings['ridge_text']:.1f}s -> {results['ridge_text']}")

    # --- b. hgb_image ------------------------------------------------------------------
    t0 = time.time()
    hgb_image = HistGradientBoostingRegressor(random_state=seed)
    hgb_image.fit(X_image_train, y_train_log)
    pred_log = hgb_image.predict(X_image_val)
    pred_price = to_price(pred_log)
    results["hgb_image"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    val_preds_price["hgb_image"] = pred_price
    timings["hgb_image"] = time.time() - t0
    joblib.dump(hgb_image, models_dir / "hgb_image.joblib")
    print(f"[train] hgb_image done in {timings['hgb_image']:.1f}s -> {results['hgb_image']}")

    # --- c. hgb_text ---------------------------------------------------------------------
    t0 = time.time()
    hgb_text = HistGradientBoostingRegressor(random_state=seed)
    hgb_text.fit(X_text_train, y_train_log)
    pred_log = hgb_text.predict(X_text_val)
    pred_price = to_price(pred_log)
    results["hgb_text"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    val_preds_price["hgb_text"] = pred_price
    timings["hgb_text"] = time.time() - t0
    joblib.dump(hgb_text, models_dir / "hgb_text.joblib")
    print(f"[train] hgb_text done in {timings['hgb_text']:.1f}s -> {results['hgb_text']}")

    # --- d. hgb_fused ---------------------------------------------------------------------
    t0 = time.time()
    hgb_fused = HistGradientBoostingRegressor(random_state=seed)
    hgb_fused.fit(X_fused_train, y_train_log)
    pred_log = hgb_fused.predict(X_fused_val)
    pred_price = to_price(pred_log)
    results["hgb_fused"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    val_preds_price["hgb_fused"] = pred_price
    timings["hgb_fused"] = time.time() - t0
    joblib.dump(hgb_fused, models_dir / "hgb_fused.joblib")
    print(f"[train] hgb_fused done in {timings['hgb_fused']:.1f}s -> {results['hgb_fused']}")

    # --- e. binres_fused ---------------------------------------------------------------------
    t0 = time.time()
    binres_clf, binres_reg, pred_log, binres_extras = train_binres_fused(
        X_fused_train, X_fused_val, train_bins, y_train_log, n_bins, seed
    )
    pred_price = to_price(pred_log)
    results["binres_fused"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    val_preds_price["binres_fused"] = pred_price
    timings["binres_fused"] = time.time() - t0
    joblib.dump(binres_clf, models_dir / "binres_classifier.joblib")
    joblib.dump(binres_reg, models_dir / "binres_regressor.joblib")
    with open(models_dir / "binres_bin_stats.json", "w", encoding="utf-8") as f:
        json.dump(binres_extras, f, indent=2)
    print(f"[train] binres_fused done in {timings['binres_fused']:.1f}s -> {results['binres_fused']}")

    # --- f. mlp_fused ---------------------------------------------------------------------
    t0 = time.time()
    mlp_model, pred_log, mlp_meta = train_mlp_fused(
        cfg["mlp"], X_fused_train, y_train_log, X_fused_val, y_val_log, y_val_price, seed
    )
    pred_price = to_price(pred_log)
    results["mlp_fused"] = compute_metrics(y_val_price, pred_price, y_val_log, pred_log)
    results["mlp_fused"]["best_epoch"] = mlp_meta["best_epoch"]
    results["mlp_fused"]["epochs_trained"] = mlp_meta["epochs_trained"]
    val_preds_price["mlp_fused"] = pred_price
    timings["mlp_fused"] = time.time() - t0
    import torch

    torch.save(mlp_model.state_dict(), models_dir / "mlp_fused.pt")
    with open(models_dir / "mlp_fused_history.json", "w", encoding="utf-8") as f:
        json.dump(mlp_meta, f, indent=2)
    print(f"[train] mlp_fused done in {timings['mlp_fused']:.1f}s -> {results['mlp_fused']}")

    # --- save scalers -----------------------------------------------------------------------
    joblib.dump(scaler_text, models_dir / "scaler_text.joblib")
    joblib.dump(scaler_image, models_dir / "scaler_image.joblib")
    joblib.dump(scaler_fused, models_dir / "scaler_fused.joblib")

    # --- pick best model by val SMAPE ------------------------------------------------------
    best_model = min(results.keys(), key=lambda k: results[k]["smape"])
    print(f"[train] best model by val SMAPE: {best_model} (SMAPE={results[best_model]['smape']:.2f})")

    # --- predictions_val.csv -----------------------------------------------------------------
    pred_df = pd.DataFrame({"sample_id": val_sample_ids, "y_true": y_val_price})
    for name, arr in val_preds_price.items():
        pred_df[f"y_pred_{name}"] = arr
    pred_df["y_pred_best"] = pred_df[f"y_pred_{best_model}"]
    pred_df["best_model"] = best_model
    pred_csv_path = artifacts_dir / "predictions_val.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[train] saved predictions -> {pred_csv_path}")

    # --- per-decile SMAPE for the best model -------------------------------------------------
    per_decile = per_decile_smape(y_val_price, val_preds_price[best_model])

    total_elapsed = time.time() - t_start
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "seed": seed,
            "subset_size": splits_info["subset_size_used"],
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_bins": n_bins,
            "val_fraction": cfg["val_fraction"],
            "image_model": cfg["image_model"],
            "text_model": cfg["text_model"],
            "image_emb_dim": int(emb_image.shape[1]),
            "text_emb_dim": int(emb_text.shape[1]),
            "tabular_feature_count": len(tabular_cols),
            "tabular_feature_columns": tabular_cols,
            "fused_dim": int(X_fused.shape[1]),
        },
        "models": results,
        "best_model": best_model,
        "per_decile": per_decile,
        "timings_sec": timings,
        "total_train_elapsed_sec": total_elapsed,
    }
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] saved metrics -> {metrics_path}")
    print(f"[train] total elapsed: {total_elapsed:.1f}s")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all model heads and evaluate on the val split.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
