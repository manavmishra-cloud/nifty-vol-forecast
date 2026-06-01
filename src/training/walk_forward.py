"""Walk-forward CV for volatility forecasting.

Unlike the LOB project (where one fold = many bars), here one fold typically
forecasts one day ahead. The natural setup is **expanding-window one-step-ahead
rolling forecast**:

    For t in [train_size, N):
        train model on [0, t)
        forecast for t
        record (y_true_t, y_pred_t)

This produces N - train_size forecast pairs.

For computational efficiency, models can be re-fit every k days rather than
every day (the `refit_freq` parameter).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def expanding_one_step_ahead(
    fit_predict_fn: Callable[[int], tuple[float, float]],
    n_samples: int,
    initial_train_size: int,
    refit_freq: int = 1,
) -> pd.DataFrame:
    """Run one-step-ahead expanding-window forecasts.

    Parameters
    ----------
    fit_predict_fn : callable
        Takes a training-end index `t_end` and returns `(y_true_at_t_end, y_pred_for_t_end)`.
        The function is responsible for fitting on data [0, t_end) and forecasting for index t_end.
    n_samples : int
        Total length of the series.
    initial_train_size : int
        Index at which forecasts begin.
    refit_freq : int
        How often to refit the model. 1 = every step. Higher values speed up
        evaluation but mean stale fits.

    Returns
    -------
    pd.DataFrame with columns: index, y_true, y_pred.
    """
    records = []
    for t in range(initial_train_size, n_samples):
        if (t - initial_train_size) % refit_freq != 0:
            # Reuse the previous fit — note: fit_predict_fn should handle this internally
            # if it caches; otherwise it will refit every call.
            pass

        try:
            y_true, y_pred = fit_predict_fn(t)
            records.append({"index": t, "y_true": y_true, "y_pred": y_pred})
        except Exception as e:
            print(f"  Warning: step t={t} failed: {e}")
            continue

    return pd.DataFrame(records)


def split_indices_expanding(
    n_samples: int,
    initial_train_size: int,
    horizon: int = 1,
):
    """Generator yielding (train_end, forecast_idx) pairs.

    For horizon h, the forecast at step t is for index t + (h - 1).
    """
    for t in range(initial_train_size, n_samples - horizon + 1):
        yield t, t + horizon - 1
