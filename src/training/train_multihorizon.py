r"""Multi-horizon volatility forecasting benchmark.

Extends the main 1-day-ahead comparison to 5-day and 22-day horizons,
running all five models (HAR-RV, GARCH, XGBoost, LSTM, Transformer)
at each horizon under the same walk-forward CV setup.

Convention: at horizon $h$, the target is $\log \mathrm{RV}_{t+h}$
(single day's variance, $h$ days ahead). This matches the direct
multi-step forecasting convention used in most of the volatility
forecasting literature.

Usage:
    python -m src.training.train_multihorizon \
        --horizons 5 22 \
        --models har garch xgb lstm transformer \
        --initial-train-size 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loaders import load_nifty
from src.data.realized_variance import add_realized_variance_columns
from src.models.ml_models import XGBoostVolForecaster, build_volatility_features
from src.models.garch_models import GARCHBaseline
from src.evaluation.metrics import evaluate_forecast, diebold_mariano


# ---------------------------------------------------------------------------
# Horizon-aware HAR-RV
# ---------------------------------------------------------------------------

def fit_har_at_horizon(rv_series: pd.Series, horizon: int):
    """Fit HAR-RV with target log(RV_{t+horizon}).

    Returns (model, X_full, y_full, index_full) ready for prediction.
    """
    from sklearn.linear_model import LinearRegression

    log_rv = np.log(rv_series.replace(0, np.nan)).dropna()
    features = pd.DataFrame({
        "log_rv_d": log_rv,
        "log_rv_w": log_rv.rolling(5).mean(),
        "log_rv_m": log_rv.rolling(22).mean(),
    })
    target = log_rv.shift(-horizon)
    full = features.join(target.rename("y")).dropna()
    X = full[["log_rv_d", "log_rv_w", "log_rv_m"]].values
    y = full["y"].values

    model = LinearRegression()
    model.fit(X, y)
    return model, full


def run_har_at_horizon(df: pd.DataFrame, horizon: int, initial_train_size: int) -> pd.DataFrame:
    """Walk-forward HAR-RV at the given horizon."""
    from sklearn.linear_model import LinearRegression

    print(f"\n=== HAR-RV (horizon={horizon}) ===")
    rv = df["rv_yang_zhang"]
    log_rv = np.log(rv.replace(0, np.nan))

    features_panel = pd.DataFrame({
        "log_rv_d": log_rv,
        "log_rv_w": log_rv.rolling(5).mean(),
        "log_rv_m": log_rv.rolling(22).mean(),
    })

    records = []
    for t in range(initial_train_size, len(df) - horizon):
        # Build training data up to step t (with horizon look-ahead in target)
        train_mask = features_panel.index < df.index[t]
        train_features = features_panel[train_mask].copy()
        train_target = log_rv.shift(-horizon).loc[train_features.index]
        combined = train_features.join(train_target.rename("y")).dropna()

        if len(combined) < 50:
            continue

        model = LinearRegression()
        X_train = combined[["log_rv_d", "log_rv_w", "log_rv_m"]].values
        y_train = combined["y"].values
        model.fit(X_train, y_train)

        # Predict for time t
        row = features_panel.iloc[t : t + 1]
        if row[["log_rv_d", "log_rv_w", "log_rv_m"]].isna().any().any():
            continue
        x_pred = row[["log_rv_d", "log_rv_w", "log_rv_m"]].values
        y_pred_log = float(model.predict(x_pred)[0])
        y_pred_var = float(np.exp(y_pred_log))

        # Realized at t+horizon
        if t + horizon >= len(df):
            break
        y_true = float(rv.iloc[t + horizon])

        records.append({
            "index": t,
            "date": df.index[t + horizon],
            "y_true": y_true,
            "y_pred": y_pred_var,
        })

        if (t - initial_train_size) % 300 == 0:
            print(f"  step {t - initial_train_size} / {len(df) - initial_train_size}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Horizon-aware GARCH
# ---------------------------------------------------------------------------

def run_garch_at_horizon(df: pd.DataFrame, horizon: int, initial_train_size: int,
                         refit_freq: int = 22) -> pd.DataFrame:
    """Walk-forward GARCH(1,1) at horizon h (refit periodically for speed)."""
    print(f"\n=== GARCH(1,1) (horizon={horizon}, refit every {refit_freq} days) ===")
    returns = df["log_return"]
    rv = df["rv_yang_zhang"]
    records = []
    model = None
    last_fit_t = -1

    for t in range(initial_train_size, len(df) - horizon):
        if model is None or (t - last_fit_t) >= refit_freq:
            try:
                model = GARCHBaseline()
                model.fit(returns.iloc[:t])
                last_fit_t = t
            except Exception:
                continue

        try:
            # GARCH at horizon h: use the h-th step ahead forecast
            from arch import arch_model
            # Use the underlying _fit_result for multi-step
            fcast = model._fit_result.forecast(horizon=horizon, reindex=False)
            var_pct = fcast.variance.values[-1, horizon - 1]  # h-th step
            y_pred = float(var_pct / 1e4)  # un-scale from percent
            y_true = float(rv.iloc[t + horizon])
            records.append({
                "index": t,
                "date": df.index[t + horizon],
                "y_true": y_true,
                "y_pred": y_pred,
            })
        except Exception:
            continue

        if (t - initial_train_size) % 300 == 0:
            print(f"  step {t - initial_train_size} / {len(df) - initial_train_size}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Horizon-aware XGBoost
# ---------------------------------------------------------------------------

def run_xgb_at_horizon(df: pd.DataFrame, horizon: int, initial_train_size: int,
                       refit_freq: int = 22) -> pd.DataFrame:
    """Walk-forward XGBoost at horizon h (target shifted by -h)."""
    print(f"\n=== XGBoost (horizon={horizon}, refit every {refit_freq} days) ===")
    features = build_volatility_features(df, log_rv_col="rv_yang_zhang", return_col="log_return")
    target = np.log(df["rv_yang_zhang"].replace(0, np.nan)).shift(-horizon).rename("log_rv_target")
    combined = features.join(target).dropna()

    X_full = combined.drop(columns=["log_rv_target"])
    y_full = combined["log_rv_target"]

    records = []
    model = None
    last_fit_t = -1

    for t in range(initial_train_size, len(X_full)):
        if model is None or (t - last_fit_t) >= refit_freq:
            model = XGBoostVolForecaster()
            model.fit(X_full.iloc[:t], y_full.iloc[:t])
            last_fit_t = t

        y_pred_log = model.predict(X_full.iloc[t : t + 1])[0]
        y_pred = float(np.exp(y_pred_log))
        # Date for which we're predicting (target date)
        cur_idx = X_full.index[t]
        target_loc = df.index.get_loc(cur_idx) + horizon if cur_idx in df.index else None
        if target_loc is None or target_loc >= len(df):
            continue
        target_date = df.index[target_loc]
        y_true = float(df["rv_yang_zhang"].iloc[target_loc])
        records.append({"index": t, "date": target_date, "y_true": y_true, "y_pred": y_pred})

        if (t - initial_train_size) % 300 == 0:
            print(f"  step {t - initial_train_size} / {len(X_full) - initial_train_size}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Horizon-aware deep models (LSTM, Transformer)
# ---------------------------------------------------------------------------

def run_deep_at_horizon(df: pd.DataFrame, model_name: str, horizon: int,
                        initial_train_size: int, refit_freq: int = 250,
                        seq_len: int = 22) -> pd.DataFrame:
    """LSTM or Transformer at horizon h."""
    from src.models.deep_models import (
        LSTMForecaster, TransformerForecaster, TrainConfig,
        build_sequences, train_deep_model, predict_deep_model,
    )

    print(f"\n=== {model_name.upper()} (horizon={horizon}) ===")

    # Prepare features
    work = df.copy()
    work["log_rv"] = np.log(work["rv_yang_zhang"].replace(0, np.nan))
    work["log_rv_parkinson"] = np.log(work["rv_parkinson"].replace(0, np.nan))
    work["log_rv_garman_klass"] = np.log(work["rv_garman_klass"].replace(0, np.nan))
    feature_cols = ["log_return", "log_rv", "log_rv_parkinson", "log_rv_garman_klass"]
    target_col = "log_rv"
    work = work.dropna(subset=feature_cols + [target_col])

    # Custom sequence build for horizon h: target = log(RV) h steps after window end
    features = work[feature_cols].values
    targets = work[target_col].values
    n = len(work)
    n_samples = n - seq_len - (horizon - 1)
    if n_samples <= 0:
        return pd.DataFrame()

    X = np.empty((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    for t in range(n_samples):
        X[t] = features[t : t + seq_len]
        y[t] = targets[t + seq_len + horizon - 1]
    pred_dates = work.index[seq_len + horizon - 1 :]

    n_features = X.shape[2]
    config = TrainConfig(epochs=80, batch_size=32, learning_rate=1e-3, verbose=False)

    fitted = None
    last_fit_t = -1
    records = []
    for t in range(initial_train_size, n_samples):
        if fitted is None or (t - last_fit_t) >= refit_freq:
            if model_name == "lstm":
                model = LSTMForecaster(n_features=n_features, hidden_size=64, num_layers=2, dropout=0.2)
            else:
                model = TransformerForecaster(
                    n_features=n_features, d_model=64, nhead=4, num_layers=2,
                    dim_feedforward=128, dropout=0.2, seq_len=seq_len,
                )
            fitted = train_deep_model(model, X[:t], y[:t], config=config)
            last_fit_t = t

        y_pred_log = predict_deep_model(fitted, X[t : t + 1])[0]
        y_pred = float(np.exp(y_pred_log))
        y_true_log = y[t]
        y_true = float(np.exp(y_true_log))
        records.append({
            "index": t,
            "date": pred_dates[t],
            "y_true": y_true,
            "y_pred": y_pred,
        })

        if (t - initial_train_size) % 300 == 0:
            print(f"  step {t - initial_train_size} / {n_samples - initial_train_size}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/nifty.parquet")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 22])
    parser.add_argument("--models", nargs="+", default=["har", "garch", "xgb", "lstm", "transformer"])
    parser.add_argument("--initial-train-size", type=int, default=1000)
    parser.add_argument("--refit-freq", type=int, default=22)
    parser.add_argument("--deep-refit-freq", type=int, default=250)
    parser.add_argument("--seq-len", type=int, default=22)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    ohlcv = load_nifty(args.data)
    df = add_realized_variance_columns(ohlcv)
    df = df.dropna(subset=["rv_yang_zhang", "log_return"])
    print(f"Loaded {len(df):,} trading days")

    all_horizon_results = {}

    for h in args.horizons:
        print(f"\n{'#' * 60}\n# HORIZON {h}\n{'#' * 60}")
        horizon_dir = Path(args.results_dir) / "metrics" / f"h{h}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        horizon_results = {}
        horizon_predictions = {}

        if "har" in args.models:
            preds = run_har_at_horizon(df, h, args.initial_train_size)
            preds.to_csv(horizon_dir / "har_predictions.csv", index=False)
            horizon_predictions["har"] = preds
            horizon_results["har"] = evaluate_forecast(
                preds["y_true"].values, preds["y_pred"].values, f"har_h{h}")

        if "garch" in args.models:
            preds = run_garch_at_horizon(df, h, args.initial_train_size, args.refit_freq)
            preds.to_csv(horizon_dir / "garch_predictions.csv", index=False)
            horizon_predictions["garch"] = preds
            horizon_results["garch"] = evaluate_forecast(
                preds["y_true"].values, preds["y_pred"].values, f"garch_h{h}")

        if "xgb" in args.models:
            preds = run_xgb_at_horizon(df, h, args.initial_train_size, args.refit_freq)
            preds.to_csv(horizon_dir / "xgb_predictions.csv", index=False)
            horizon_predictions["xgb"] = preds
            horizon_results["xgb"] = evaluate_forecast(
                preds["y_true"].values, preds["y_pred"].values, f"xgb_h{h}")

        for deep_name in ("lstm", "transformer"):
            if deep_name in args.models:
                preds = run_deep_at_horizon(
                    df, deep_name, h, args.initial_train_size,
                    refit_freq=args.deep_refit_freq, seq_len=args.seq_len)
                preds.to_csv(horizon_dir / f"{deep_name}_predictions.csv", index=False)
                horizon_predictions[deep_name] = preds
                horizon_results[deep_name] = evaluate_forecast(
                    preds["y_true"].values, preds["y_pred"].values, f"{deep_name}_h{h}")

        # Diebold-Mariano vs HAR
        if "har" in horizon_predictions and len(args.models) > 1:
            print(f"\n=== DM vs HAR-RV (h={h}) ===")
            har_preds = horizon_predictions["har"]
            for name, preds in horizon_predictions.items():
                if name == "har":
                    continue
                merged = har_preds[["date", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_har"}).merge(
                    preds[["date", "y_pred"]].rename(columns={"y_pred": "y_pred_model"}),
                    on="date",
                )
                if len(merged) == 0:
                    continue
                e_har = (np.log(merged["y_pred_har"].clip(lower=1e-10)) - np.log(merged["y_true"].clip(lower=1e-10))) ** 2
                e_model = (np.log(merged["y_pred_model"].clip(lower=1e-10)) - np.log(merged["y_true"].clip(lower=1e-10))) ** 2
                dm, p = diebold_mariano(e_har.values, e_model.values, h=h)
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
                horizon_results[name]["dm_vs_har"] = dm
                horizon_results[name]["dm_pvalue"] = p
                print(f"  HAR vs {name}: DM={dm:>7.3f} (p={p:.4f}) {sig}")

        # Save horizon summary
        with open(horizon_dir / "summary.json", "w") as f:
            json.dump(horizon_results, f, indent=2, default=float)

        all_horizon_results[h] = horizon_results

    # Cross-horizon summary table
    print(f"\n{'=' * 60}\n=== Multi-horizon summary ===\n{'=' * 60}")
    print(f"{'Horizon':<10}{'Model':<15}{'QLIKE':>10}{'DM vs HAR':>12}{'p-value':>10}")
    for h, results in all_horizon_results.items():
        for name, res in results.items():
            dm = res.get("dm_vs_har", "--")
            p = res.get("dm_pvalue", "--")
            dm_str = f"{dm:.3f}" if isinstance(dm, float) else str(dm)
            p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
            print(f"{h:<10}{name:<15}{res['qlike']:>10.4f}{dm_str:>12}{p_str:>10}")

    with open(Path(args.results_dir) / "metrics" / "multihorizon_summary.json", "w") as f:
        json.dump({str(k): v for k, v in all_horizon_results.items()}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
