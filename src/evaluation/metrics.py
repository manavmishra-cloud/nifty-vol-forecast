"""Evaluation metrics for volatility forecasting.

Standard losses in the volatility literature:

- **MSE** on log-variance: penalizes large absolute errors symmetrically.
- **QLIKE** loss (Patton 2011): the gold-standard loss for vol forecasting —
  robust to noisy proxies of the true latent variance.

      QLIKE(sigma2_hat, sigma2) = sigma2 / sigma2_hat - log(sigma2 / sigma2_hat) - 1

  QLIKE penalizes underprediction more than overprediction, which matches the
  practical cost of underestimating risk.

- **Mincer-Zarnowitz regression** of realized on forecast:
      sigma2_t = alpha + beta * sigma2_hat_t + e_t
  An unbiased forecast has alpha=0, beta=1.

- **Diebold-Mariano test** for pairwise model comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mse_log_variance(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    """MSE on log-variance — symmetric in log space."""
    mask = (y_true_var > 0) & (y_pred_var > 0) & ~np.isnan(y_true_var) & ~np.isnan(y_pred_var)
    return float(np.mean((np.log(y_pred_var[mask]) - np.log(y_true_var[mask])) ** 2))


def qlike_loss(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    """QLIKE loss (Patton 2011). Lower is better."""
    mask = (y_true_var > 0) & (y_pred_var > 0) & ~np.isnan(y_true_var) & ~np.isnan(y_pred_var)
    yt = y_true_var[mask]
    yp = y_pred_var[mask]
    return float(np.mean(yt / yp - np.log(yt / yp) - 1.0))


def mincer_zarnowitz(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> dict:
    """Mincer-Zarnowitz regression for forecast unbiasedness.

    Returns dict with alpha, beta, R^2, and the joint test of (alpha=0, beta=1).
    """
    import statsmodels.api as sm

    mask = ~np.isnan(y_true_var) & ~np.isnan(y_pred_var)
    yt = y_true_var[mask]
    yp = y_pred_var[mask]

    X = sm.add_constant(yp)
    res = sm.OLS(yt, X).fit()

    return {
        "alpha": float(res.params[0]),
        "beta": float(res.params[1]),
        "r_squared": float(res.rsquared),
        "alpha_pvalue": float(res.pvalues[0]),
        "beta_pvalue": float(res.pvalues[1]),
    }


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Parameters
    ----------
    e1, e2 : array-like
        Loss series from model 1 and model 2 (e.g. squared errors).
    h : int
        Forecast horizon (used in HAC variance estimation).

    Returns
    -------
    (DM_statistic, p_value)

    Null hypothesis: e1 and e2 have equal mean.
    Positive DM stat = model 1 has *higher* loss (worse).
    """
    d = e1 - e2
    n = len(d)
    mean_d = np.mean(d)

    # Newey-West variance with h-1 lags
    var_d = np.var(d)
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        var_d += 2.0 * gamma_k

    dm_stat = mean_d / np.sqrt(var_d / n)
    # Two-sided p-value from t-distribution
    from scipy.stats import t
    p_value = 2.0 * (1.0 - t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)


def evaluate_forecast(
    y_true_var: np.ndarray,
    y_pred_var: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Compute all standard volatility forecast metrics."""
    return {
        "model": model_name,
        "n_observations": int((~np.isnan(y_true_var) & ~np.isnan(y_pred_var)).sum()),
        "mse_log_var": mse_log_variance(y_true_var, y_pred_var),
        "qlike": qlike_loss(y_true_var, y_pred_var),
        "mincer_zarnowitz": mincer_zarnowitz(y_true_var, y_pred_var),
    }
