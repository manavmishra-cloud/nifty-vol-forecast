"""Visualization utilities for NIFTY volatility forecast results.

Produces publication-quality plots:

- Forecast vs realized series (line chart) — visually show model behavior
- Per-model QLIKE bar chart
- Cross-model error distribution (boxplot)
- Pairwise scatter (forecast vs realized) with regression line
- Cumulative forecast error over time
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def plot_forecast_vs_actual(
    predictions: pd.DataFrame,
    model_name: str,
    save_path: Optional[Path] = None,
    annualize: bool = True,
):
    """Line plot of realized vs forecast annualized volatility.

    Expects `predictions` with columns: date, y_true (variance), y_pred (variance).
    """
    df = predictions.dropna(subset=["y_true", "y_pred"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if annualize:
        df["true_vol"] = np.sqrt(df["y_true"].clip(lower=0) * 252) * 100
        df["pred_vol"] = np.sqrt(df["y_pred"].clip(lower=0) * 252) * 100
        ylabel = "Annualized volatility (%)"
    else:
        df["true_vol"] = df["y_true"]
        df["pred_vol"] = df["y_pred"]
        ylabel = "Variance"

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df["date"], df["true_vol"], color="black", linewidth=0.6, alpha=0.7, label="Realized")
    ax.plot(df["date"], df["pred_vol"], color="crimson", linewidth=0.7, alpha=0.9, label=f"{model_name} forecast")
    ax.set_ylabel(ylabel)
    ax.set_title(f"NIFTY-50 realized vs {model_name} forecast (1-day ahead)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_qlike_comparison(
    summary: dict,
    save_path: Optional[Path] = None,
):
    """Bar chart of QLIKE loss across models (lower = better)."""
    models = list(summary.keys())
    qlike_values = [summary[m]["qlike"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["steelblue" if m == "har" else "lightcoral" for m in models]
    bars = ax.bar(models, qlike_values, color=colors)
    ax.set_ylabel("QLIKE loss (lower = better)")
    ax.set_title("Forecast quality across models (NIFTY-50)")

    # Add value labels
    for bar, val in zip(bars, qlike_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_scatter_forecast_vs_realized(
    predictions: pd.DataFrame,
    model_name: str,
    save_path: Optional[Path] = None,
):
    """Scatter of forecast vs realized (Mincer-Zarnowitz visual).

    A perfect forecaster falls on the 45-degree line.
    """
    df = predictions.dropna(subset=["y_true", "y_pred"]).copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_pred"], df["y_true"], s=4, alpha=0.3, color="steelblue")

    lim = max(df["y_true"].max(), df["y_pred"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel(f"{model_name} forecast (variance)")
    ax.set_ylabel("Realized variance")
    ax.set_title(f"{model_name}: forecast vs realized")
    ax.legend()
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_cumulative_loss_over_time(
    predictions_dict: dict,
    save_path: Optional[Path] = None,
):
    """Cumulative QLIKE over time, one line per model.

    Useful for seeing whether one model gains advantage in specific periods
    (e.g., crisis windows) vs steady-state performance.
    """
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for name, preds in predictions_dict.items():
        df = preds.dropna(subset=["y_true", "y_pred"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # Per-row QLIKE
        df["qlike"] = df["y_true"] / df["y_pred"] - np.log(df["y_true"] / df["y_pred"]) - 1.0
        df["cum_qlike"] = df["qlike"].cumsum()
        ax.plot(df["date"], df["cum_qlike"], label=name, linewidth=0.9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative QLIKE loss")
    ax.set_title("Cumulative forecast error by model (lower = better)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig, ax
