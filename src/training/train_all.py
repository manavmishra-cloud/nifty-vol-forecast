"""End-to-end training pipeline for volatility forecasting baselines.

Usage:
    # 1. First time: download data
    python -m src.data.loaders --download

    # 2. Run all baseline models
    python -m src.training.train_all \\
        --data data/processed/nifty.parquet \\
        --initial-train-size 1000 \\
        --horizon 1 \\
        --models har xgb

Outputs:
    - results/metrics/{model}_predictions.csv
    - results/metrics/all_models_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loaders import download_nifty_yfinance, load_nifty
from src.data.realized_variance import add_realized_variance_columns
from src.models.garch_models import GARCHBaseline, EGARCHBaseline, HARRVBaseline
from src.models.ml_models import XGBoostVolForecaster, build_volatility_features
from src.evaluation.metrics import evaluate_forecast, diebold_mariano


def prepare_data(data_path: Optional[Path] = None, download: bool = False) -> pd.DataFrame:
    """Load or download NIFTY data and compute realized variance columns."""
    if download or not data_path or not data_path.exists():
        print("Downloading NIFTY data from yfinance...")
        save_path = data_path or Path("data/processed/nifty.parquet")
        ohlcv = download_nifty_yfinance(start="2007-01-01", save_path=save_path)
        print(f"  Saved to {save_path}")
    else:
        print(f"Loading NIFTY data from {data_path}")
        ohlcv = load_nifty(data_path)

    df = add_realized_variance_columns(ohlcv)
    df = df.dropna(subset=["rv_yang_zhang", "log_return"])
    print(f"  {len(df):,} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def run_har_rv(df: pd.DataFrame, initial_train_size: int) -> pd.DataFrame:
    """HAR-RV walk-forward: refit every step (cheap)."""
    print("\n=== HAR-RV ===")
    rv = df["rv_yang_zhang"]
    records = []
    for t in range(initial_train_size, len(df) - 1):
        model = HARRVBaseline()
        try:
            model.fit(rv.iloc[:t])
            y_pred = model.forecast_variance(rv.iloc[:t])
            y_true = rv.iloc[t]
            records.append({"index": t, "date": df.index[t], "y_true": y_true, "y_pred": y_pred})
        except Exception as e:
            continue
        if (t - initial_train_size) % 200 == 0:
            print(f"  step {t - initial_train_size} / {len(df) - initial_train_size}")
    return pd.DataFrame(records)


def run_garch(df: pd.DataFrame, initial_train_size: int, refit_freq: int = 22) -> pd.DataFrame:
    """GARCH(1,1) walk-forward — refit monthly for speed."""
    print(f"\n=== GARCH(1,1) (refit every {refit_freq} days) ===")
    returns = df["log_return"]
    records = []
    model = None
    last_fit_t = -1

    for t in range(initial_train_size, len(df) - 1):
        if model is None or (t - last_fit_t) >= refit_freq:
            model = GARCHBaseline()
            try:
                model.fit(returns.iloc[:t])
                last_fit_t = t
            except Exception as e:
                continue

        try:
            y_pred = model.forecast_variance(horizon=1)
            y_true = df["rv_yang_zhang"].iloc[t]
            records.append({"index": t, "date": df.index[t], "y_true": y_true, "y_pred": y_pred})
        except Exception:
            continue

        if (t - initial_train_size) % 200 == 0:
            print(f"  step {t - initial_train_size} / {len(df) - initial_train_size}")
    return pd.DataFrame(records)


def run_xgboost(df: pd.DataFrame, initial_train_size: int, refit_freq: int = 22) -> pd.DataFrame:
    """XGBoost walk-forward — refit periodically."""
    print(f"\n=== XGBoost (refit every {refit_freq} days) ===")
    features = build_volatility_features(df, log_rv_col="rv_yang_zhang", return_col="log_return")
    target = np.log(df["rv_yang_zhang"].replace(0, np.nan)).shift(-1).rename("log_rv_next")

    combined = features.join(target).dropna()
    X_full = combined.drop(columns=["log_rv_next"])
    y_full = combined["log_rv_next"]

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
        # True forward-looking RV
        actual_idx = combined.index[t]
        y_true = float(df.loc[actual_idx, "rv_yang_zhang"]) if actual_idx in df.index else float("nan")
        records.append({"index": t, "date": actual_idx, "y_true": y_true, "y_pred": y_pred})

        if (t - initial_train_size) % 200 == 0:
            print(f"  step {t - initial_train_size} / {len(X_full) - initial_train_size}")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/nifty.parquet")
    parser.add_argument("--download", action="store_true", help="Force re-download of NIFTY data")
    parser.add_argument("--initial-train-size", type=int, default=1000, help="Trading days for initial training (~4 yrs)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days")
    parser.add_argument("--models", nargs="+", default=["har", "garch", "xgb"], choices=["har", "garch", "egarch", "xgb"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--refit-freq", type=int, default=22, help="Days between refits for GARCH/XGB")
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = prepare_data(data_path=data_path, download=args.download)

    all_results = {}
    all_predictions = {}

    if "har" in args.models:
        preds = run_har_rv(df, initial_train_size=args.initial_train_size)
        preds.to_csv(results_dir / "metrics" / "har_predictions.csv", index=False)
        all_predictions["har"] = preds
        all_results["har"] = evaluate_forecast(preds["y_true"].values, preds["y_pred"].values, "har")

    if "garch" in args.models:
        preds = run_garch(df, initial_train_size=args.initial_train_size, refit_freq=args.refit_freq)
        preds.to_csv(results_dir / "metrics" / "garch_predictions.csv", index=False)
        all_predictions["garch"] = preds
        all_results["garch"] = evaluate_forecast(preds["y_true"].values, preds["y_pred"].values, "garch")

    if "xgb" in args.models:
        preds = run_xgboost(df, initial_train_size=args.initial_train_size, refit_freq=args.refit_freq)
        preds.to_csv(results_dir / "metrics" / "xgb_predictions.csv", index=False)
        all_predictions["xgb"] = preds
        all_results["xgb"] = evaluate_forecast(preds["y_true"].values, preds["y_pred"].values, "xgb")

    # Cross-model summary
    summary_path = results_dir / "metrics" / "all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n=== Summary ===")
    print(f"{'Model':<10} {'QLIKE':>12} {'MSE log-var':>14} {'N obs':>8}")
    for name, res in all_results.items():
        print(f"{name:<10} {res['qlike']:>12.6f} {res['mse_log_var']:>14.6f} {res['n_observations']:>8}")

    # Pairwise Diebold-Mariano vs HAR (the standard benchmark)
    if "har" in all_predictions and len(args.models) > 1:
        print("\n=== Diebold-Mariano vs HAR-RV ===")
        har_preds = all_predictions["har"]
        for name, preds in all_predictions.items():
            if name == "har":
                continue
            # Align on date
            merged = har_preds[["date", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_har"}).merge(
                preds[["date", "y_pred"]].rename(columns={"y_pred": "y_pred_model"}),
                on="date",
            )
            e_har = (np.log(merged["y_pred_har"].clip(lower=1e-10)) - np.log(merged["y_true"].clip(lower=1e-10))) ** 2
            e_model = (np.log(merged["y_pred_model"].clip(lower=1e-10)) - np.log(merged["y_true"].clip(lower=1e-10))) ** 2
            dm, p = diebold_mariano(e_har.values, e_model.values, h=args.horizon)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
            print(f"  HAR vs {name}: DM={dm:>7.3f} (p={p:.4f}) {sig}")


if __name__ == "__main__":
    main()
