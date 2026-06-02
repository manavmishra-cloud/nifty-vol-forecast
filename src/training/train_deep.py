"""Walk-forward training pipeline for deep sequence models (LSTM, Transformer).

Unlike the GARCH/HAR/XGBoost models which can be refit cheaply every step,
deep models are expensive to retrain. We use periodic retraining: train once
on the initial window, then refit every `refit_freq` days (default ~250 ≈
once per trading year). Between refits we use the most recent trained model
for one-step-ahead prediction.

Usage (called from train_all.py with --models lstm transformer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.models.deep_models import (
    LSTMForecaster,
    TransformerForecaster,
    TrainConfig,
    build_sequences,
    train_deep_model,
    predict_deep_model,
)


# Default feature columns used as input to deep models.
# We pass raw daily features — the sequence carries the temporal info.
DEFAULT_FEATURE_COLS = ["log_return", "log_rv", "rv_parkinson", "rv_garman_klass"]


def _prepare_deep_dataset(df: pd.DataFrame, seq_len: int = 22) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Build (X, y, dates, feature_names) from a NIFTY OHLCV+RV DataFrame.

    Target: log(rv_yang_zhang) at t+1.
    Features: log_return, log of each realized-variance estimator.
    """
    work = df.copy()
    work["log_rv"] = np.log(work["rv_yang_zhang"].replace(0, np.nan))
    # Use log of RV estimators as additional features (avoid raw scale issues)
    work["log_rv_parkinson"] = np.log(work["rv_parkinson"].replace(0, np.nan))
    work["log_rv_garman_klass"] = np.log(work["rv_garman_klass"].replace(0, np.nan))

    feature_cols = ["log_return", "log_rv", "log_rv_parkinson", "log_rv_garman_klass"]
    target_col = "log_rv"

    work = work.dropna(subset=feature_cols + [target_col])

    X, y, dates = build_sequences(work, feature_cols, target_col, seq_len=seq_len)
    return X, y, dates, feature_cols


def run_deep_model(
    df: pd.DataFrame,
    model_name: str,
    initial_train_size: int,
    refit_freq: int = 250,
    seq_len: int = 22,
    verbose: bool = False,
) -> pd.DataFrame:
    """Walk-forward training for LSTM or Transformer.

    Parameters
    ----------
    df : pd.DataFrame
        NIFTY data with realized variance columns already computed.
    model_name : str
        "lstm" or "transformer".
    initial_train_size : int
        Number of trading days for the first training window.
    refit_freq : int
        How often (in days) to retrain the model. ~250 = once per trading year.
    seq_len : int
        Sequence length (lookback window).
    verbose : bool
        Print progress.

    Returns
    -------
    DataFrame with columns: index, date, y_true (log RV), y_pred (log RV).

    Note: predictions and targets are in *log realized variance* space.
    The caller should exponentiate when comparing against y_true_var from other models.
    """
    print(f"\n=== {model_name.upper()} ===")

    X, y, dates, feature_cols = _prepare_deep_dataset(df, seq_len=seq_len)
    n_features = X.shape[2]
    n_samples = len(X)

    print(f"  Sequences: {n_samples:,} × {seq_len} × {n_features}")
    print(f"  Initial train: first {initial_train_size:,} samples")
    print(f"  Refit every {refit_freq} days")

    # The walk-forward setup: at each step t (>= initial_train_size), we
    # predict y[t] using a model that was trained on samples [0, last_refit_t).
    # We retrain when t crosses each refit boundary.

    config = TrainConfig(epochs=80, batch_size=32, learning_rate=1e-3, verbose=verbose)

    fitted = None
    last_fit_t = -1
    records = []

    for t in range(initial_train_size, n_samples):
        # Retrain if needed
        if fitted is None or (t - last_fit_t) >= refit_freq:
            if model_name == "lstm":
                model = LSTMForecaster(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
            elif model_name == "transformer":
                model = TransformerForecaster(
                    n_features=n_features, d_model=64, nhead=4, num_layers=2,
                    dim_feedforward=128, dropout=0.2, seq_len=seq_len,
                )
            else:
                raise ValueError(f"Unknown model_name {model_name}")

            X_train = X[:t]
            y_train = y[:t]
            fitted = train_deep_model(model, X_train, y_train, config=config)
            last_fit_t = t
            if verbose:
                print(f"  refit at t={t}")

        # Predict for current step
        y_pred_log_rv = predict_deep_model(fitted, X[t : t + 1])[0]
        y_true_log_rv = y[t]
        records.append({
            "index": t,
            "date": dates[t],
            "y_true": y_true_log_rv,    # log realized variance
            "y_pred": y_pred_log_rv,    # log realized variance
        })

        if (t - initial_train_size) % 200 == 0:
            print(f"  step {t - initial_train_size} / {n_samples - initial_train_size}")

    return pd.DataFrame(records)


def deep_predictions_to_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Convert LSTM/Transformer outputs (log RV) to variance for cross-model comparison."""
    out = df.copy()
    out["y_true_var"] = np.exp(out["y_true"])
    out["y_pred_var"] = np.exp(out["y_pred"])
    return out
