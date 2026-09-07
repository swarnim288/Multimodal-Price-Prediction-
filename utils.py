"""Shared utilities for the multimodal price-prediction pipeline.

Provides reproducibility helpers, the two custom regression metrics used
throughout the project (SMAPE, RMSLE), and a config loader that resolves
every relative path in ``config/config.yaml`` against the repository root
so scripts behave the same regardless of the working directory they are
invoked from.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

# Repository root = parent of the `src/` directory that contains this file.
REPO_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int = 42) -> None:
    """Seed python's ``random``, ``numpy``, and ``torch`` (if importable).

    Safe to call even when torch is not installed or has no CUDA device.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU box in practice
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: Union[str, Path, None] = None) -> dict[str, Any]:
    """Load ``config/config.yaml`` (or a path override) as a plain dict.

    The returned dict gains a ``_repo_root`` string entry pointing at the
    resolved repository root, which downstream code can use together with
    :func:`resolve_path` to turn the config's relative paths into absolute
    ``Path`` objects.
    """
    if path is None:
        path = REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``cfg[key]`` (a repo-root-relative path string) to an absolute Path."""
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    return (root / cfg[key]).resolve()


_TEXT_COLUMNS_MAY_BE_EMPTY = ("item_name", "unit", "details")


def load_prepared_table(cfg: dict[str, Any]) -> "pd.DataFrame":  # noqa: F821 - pandas imported lazily
    """Load ``artifacts/prepared.{parquet,csv}`` written by ``prepare_data.py``.

    Prefers the parquet file if present, else falls back to CSV. Repairs the
    well-known CSV round-trip artifact where empty strings in text columns
    (``item_name``, ``unit``, ``details``) get read back as NaN by pandas --
    those are restored to ``""`` so downstream string concatenation (e.g. in
    ``extract_embeddings.py``) never has to deal with float NaN in a text
    field. Row order on disk is the canonical order (sorted by sample_id);
    callers should not re-sort or shuffle it, since embedding ``.npy`` files
    are aligned to this table by row position.
    """
    import pandas as pd

    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    parquet_path = artifacts_dir / "prepared.parquet"
    csv_path = artifacts_dir / "prepared.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} exists. Run src/prepare_data.py first."
        )

    for col in _TEXT_COLUMNS_MAY_BE_EMPTY:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error, as a percentage in [0, 200].

    SMAPE = mean( 2*|pred - true| / (|true| + |pred|) ) * 100
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error. Predictions are clipped at 0 before log1p."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), a_min=0.0, a_max=None)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))


if __name__ == "__main__":
    # Tiny smoke test.
    seed_everything(42)
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])
    print("smape:", smape(yt, yp))
    print("rmsle:", rmsle(yt, yp))
    cfg = load_config()
    print("repo root:", cfg["_repo_root"])
    print("csv_path resolves to:", resolve_path(cfg, "csv_path"))
