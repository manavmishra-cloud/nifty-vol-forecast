"""Realized variance estimators.

When daily intraday data is available, realized variance is the sum of squared
intraday returns — an *observable* proxy for the latent variance process. With
only daily OHLC data, we use **range-based estimators** that approximate
realized variance from the high-low-open-close path.

Estimators implemented:

- **Parkinson (1980)** — uses high-low range. Assumes no drift, no jumps.
- **Garman-Klass (1980)** — uses high-low + open-close. More efficient.
- **Rogers-Satchell (1991)** — drift-robust.
- **Yang-Zhang (2000)** — combines overnight + intraday variance, robust to jumps.

For NIFTY where we typically have OHLC daily, Yang-Zhang is the strongest
estimator. Garman-Klass is the standard for academic comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson estimator: variance from high-low range.

        sigma^2 = (1 / (4 ln 2)) * (ln(High / Low))^2
    """
    log_hl = np.log(high / low)
    return (1.0 / (4.0 * np.log(2.0))) * log_hl ** 2


def garman_klass_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Garman-Klass estimator.

        sigma^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    return 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2


def rogers_satchell_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Rogers-Satchell estimator: drift-robust.

        sigma^2 = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
    """
    return (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )


def yang_zhang_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: float = 0.34,
) -> pd.Series:
    """Yang-Zhang estimator: combines overnight, opening, and intraday variance.

        sigma_YZ^2 = sigma_overnight^2 + k * sigma_opening^2 + (1-k) * sigma_RS^2

    Where sigma_overnight^2 = (ln(O_t / C_{t-1}))^2,
          sigma_opening^2   = (ln(C_t / O_t))^2,
          sigma_RS^2 = Rogers-Satchell.

    The constant k is set to 0.34 in the original paper as a near-optimal value
    for typical equity index data.
    """
    log_oc_prev = np.log(open_ / close.shift(1))  # overnight return
    log_co = np.log(close / open_)                # opening (intraday) return
    rs = rogers_satchell_variance(open_, high, low, close)

    sigma_o = log_oc_prev ** 2
    sigma_c = log_co ** 2

    return sigma_o + k * sigma_c + (1.0 - k) * rs


def add_realized_variance_columns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Append common realized-variance columns to an OHLCV DataFrame.

    Adds: rv_parkinson, rv_garman_klass, rv_rogers_satchell, rv_yang_zhang,
          log_return, sq_return.
    """
    df = ohlcv.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    df["log_return"] = np.log(c / c.shift(1))
    df["sq_return"] = df["log_return"] ** 2
    df["rv_parkinson"] = parkinson_variance(h, l)
    df["rv_garman_klass"] = garman_klass_variance(o, h, l, c)
    df["rv_rogers_satchell"] = rogers_satchell_variance(o, h, l, c)
    df["rv_yang_zhang"] = yang_zhang_variance(o, h, l, c)

    # Realized volatility = sqrt(realized variance), annualized (252 trading days)
    df["rv_yz_annualized"] = np.sqrt(df["rv_yang_zhang"] * 252.0)

    return df
