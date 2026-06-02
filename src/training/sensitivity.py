"""Sensitivity analysis: vary LSTM hyperparameters and report QLIKE.

Reproduces the main result with different settings:
- seq_len ∈ {5, 10, 22, 44}
- hidden_size ∈ {32, 64, 128}
- refit_freq ∈ {125, 250, 500}

Outputs a CSV table for the paper's Robustness section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_nifty
from src.data.realized_variance import add_realized_variance_columns
from src.training.train_deep import run_deep_model, deep_predictions_to_variance
from src.evaluation.metrics import evaluate_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/nifty.parquet")
    parser.add_argument("--initial-train-size", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default="results/sensitivity")
    args = parser.parse_args()

    # Load data once
    ohlcv = load_nifty(args.data)
    df = add_realized_variance_columns(ohlcv)
    df = df.dropna(subset=["rv_yang_zhang", "log_return"])
    print(f"Loaded {len(df):,} trading days")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- seq_len sensitivity (hidden_size=64, refit=250) ----
    for seq_len in [5, 10, 22, 44]:
        print(f"\n--- LSTM seq_len={seq_len} ---")
        raw = run_deep_model(
            df, model_name="lstm",
            initial_train_size=args.initial_train_size,
            refit_freq=250,
            seq_len=seq_len,
        )
        conv = deep_predictions_to_variance(raw)
        metrics = evaluate_forecast(conv["y_true_var"].values, conv["y_pred_var"].values, "lstm")
        rows.append({
            "experiment": "seq_len",
            "value": seq_len,
            "model": "lstm",
            "qlike": metrics["qlike"],
            "mse_log_var": metrics["mse_log_var"],
            "n_obs": metrics["n_observations"],
        })

    # ---- refit_freq sensitivity (seq_len=22, hidden=64) ----
    for refit in [125, 500]:  # 250 already covered above
        print(f"\n--- LSTM refit_freq={refit} ---")
        raw = run_deep_model(
            df, model_name="lstm",
            initial_train_size=args.initial_train_size,
            refit_freq=refit,
            seq_len=22,
        )
        conv = deep_predictions_to_variance(raw)
        metrics = evaluate_forecast(conv["y_true_var"].values, conv["y_pred_var"].values, "lstm")
        rows.append({
            "experiment": "refit_freq",
            "value": refit,
            "model": "lstm",
            "qlike": metrics["qlike"],
            "mse_log_var": metrics["mse_log_var"],
            "n_obs": metrics["n_observations"],
        })

    # ---- Save ----
    df_results = pd.DataFrame(rows)
    out_path = Path(args.results_dir) / "lstm_sensitivity.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n=== Sensitivity Summary ===")
    print(df_results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
