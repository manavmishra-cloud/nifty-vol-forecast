"""Diebold-Mariano test computed using QLIKE losses directly.

In the main pipeline we computed DM using squared log forecast errors
(MSE-log-variance loss). For longer horizons we observe that the
QLIKE ranking and the MSE-log-var ranking can disagree -- a known
issue when forecast errors have heavy or asymmetric tails. To remain
consistent with our primary loss (QLIKE), this module recomputes
DM statistics using per-observation QLIKE losses.

Usage:
    python -m src.evaluation.dm_qlike
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import diebold_mariano


def per_observation_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE losses (Patton 2011)."""
    mask = (y_true > 0) & (y_pred > 0) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    losses = np.full_like(y_true, np.nan, dtype=np.float64)
    ratio = y_true[mask] / y_pred[mask]
    losses[mask] = ratio - np.log(ratio) - 1.0
    return losses


def dm_test_qlike(predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, h: int = 1):
    """Pairwise DM test on QLIKE losses between two models.

    Both prediction DataFrames must have columns: date, y_true, y_pred.
    """
    merged = predictions_a[["date", "y_true", "y_pred"]].rename(
        columns={"y_pred": "y_pred_a"}
    ).merge(
        predictions_b[["date", "y_pred"]].rename(columns={"y_pred": "y_pred_b"}),
        on="date",
    )
    if len(merged) == 0:
        return float("nan"), float("nan")

    yt = merged["y_true"].values
    la = per_observation_qlike(yt, merged["y_pred_a"].values)
    lb = per_observation_qlike(yt, merged["y_pred_b"].values)

    valid = ~np.isnan(la) & ~np.isnan(lb)
    return diebold_mariano(la[valid], lb[valid], h=h)


def main():
    results_dir = Path("results/metrics")
    rows = []

    for h in (1, 5, 22):
        if h == 1:
            base_dir = results_dir
        else:
            base_dir = results_dir / f"h{h}"

        if not (base_dir / "har_predictions.csv").exists():
            print(f"Skipping h={h}: no HAR predictions in {base_dir}")
            continue

        har_preds = pd.read_csv(base_dir / "har_predictions.csv", parse_dates=["date"])

        for model in ("garch", "xgb", "lstm", "transformer"):
            path = base_dir / f"{model}_predictions.csv"
            if not path.exists():
                continue
            preds = pd.read_csv(path, parse_dates=["date"])
            dm, p = dm_test_qlike(har_preds, preds, h=h)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
            rows.append({
                "horizon": h,
                "model": model,
                "dm_qlike_vs_har": dm,
                "p_value": p,
                "sig": sig,
            })

    df = pd.DataFrame(rows)
    print("\n=== Diebold-Mariano on QLIKE losses (vs HAR-RV baseline) ===")
    print(df.to_string(index=False))

    out_path = results_dir / "dm_qlike_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
