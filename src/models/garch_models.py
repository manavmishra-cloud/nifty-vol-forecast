"""GARCH-family volatility models.

Wrappers around the `arch` package to provide a consistent fit/predict
interface that works with the walk-forward CV harness.

Models implemented:

- **GARCH(1,1)** — Bollerslev (1986). The standard.
- **EGARCH(1,1,1)** — Nelson (1991). Asymmetric leverage.
- **Realized GARCH** — Hansen, Huang & Shek (2012). Couples returns with a
  realized measure (so we feed it a realized-variance proxy).

All models expose:
    .fit(returns_train, rv_train=None)
    .forecast(horizon=1) -> pd.Series

Where `returns_train` are daily log returns (in percent — `arch` expects %),
and `rv_train` is an optional realized-variance series.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class GARCHBaseline:
    """GARCH(1,1) with normal innovations.

    Note: the `arch` package expects returns in **percent** (not fractional)
    for numerical stability. We rescale internally.
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self._model = None
        self._fit_result = None

    def fit(self, returns: pd.Series) -> "GARCHBaseline":
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("arch package required. `pip install arch`.") from e

        # Convert fractional returns to percent
        ret_pct = returns.dropna() * 100.0
        self._model = arch_model(ret_pct, vol="Garch", p=self.p, q=self.q, dist="normal", mean="constant")
        self._fit_result = self._model.fit(disp="off")
        return self

    def forecast_variance(self, horizon: int = 1) -> float:
        """Return forecast of variance (in fractional terms, not percent)."""
        if self._fit_result is None:
            raise RuntimeError("Call fit() first.")
        fcast = self._fit_result.forecast(horizon=horizon, reindex=False)
        # arch returns variance in (percent)^2 — convert back to fractional^2
        var_pct = fcast.variance.values[-1, -1]
        return float(var_pct / 1e4)

    def get_params(self) -> dict:
        if self._fit_result is None:
            return {}
        return dict(self._fit_result.params)


class EGARCHBaseline:
    """EGARCH(p, o, q) — captures the leverage effect.

    Default specification EGARCH(1,1,1) matches Nelson (1991).
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 1):
        self.p = p
        self.o = o
        self.q = q
        self._fit_result = None

    def fit(self, returns: pd.Series) -> "EGARCHBaseline":
        from arch import arch_model
        ret_pct = returns.dropna() * 100.0
        model = arch_model(ret_pct, vol="EGARCH", p=self.p, o=self.o, q=self.q, dist="normal", mean="constant")
        self._fit_result = model.fit(disp="off")
        return self

    def forecast_variance(self, horizon: int = 1) -> float:
        fcast = self._fit_result.forecast(horizon=horizon, reindex=False)
        var_pct = fcast.variance.values[-1, -1]
        return float(var_pct / 1e4)


class HARRVBaseline:
    """Heterogeneous Autoregressive on Realized Variance (Corsi 2009).

    Regresses log(RV_{t+1}) on log(RV) over three horizons:
        log(RV_{t+1}) = c + b_d * log(RV_t) + b_w * log(RV_t^w) + b_m * log(RV_t^m) + e

    Where:
        RV_t^w = mean of RV over last 5 days
        RV_t^m = mean of RV over last 22 days

    This is the standard benchmark in modern volatility forecasting papers and
    is notoriously hard to beat.
    """

    def __init__(self):
        self.coefficients = None

    def _build_features(self, rv: pd.Series) -> pd.DataFrame:
        log_rv = np.log(rv.replace(0, np.nan)).dropna()
        df = pd.DataFrame({
            "log_rv_d": log_rv,
            "log_rv_w": log_rv.rolling(5).mean(),
            "log_rv_m": log_rv.rolling(22).mean(),
        })
        return df

    def fit(self, rv: pd.Series) -> "HARRVBaseline":
        """Fit on a realized-variance series (length N gives N - 22 training samples)."""
        from sklearn.linear_model import LinearRegression

        features = self._build_features(rv)
        y = np.log(rv.shift(-1).replace(0, np.nan))

        df = features.join(y.rename("y")).dropna()
        X = df[["log_rv_d", "log_rv_w", "log_rv_m"]].values
        y_train = df["y"].values

        model = LinearRegression()
        model.fit(X, y_train)
        self.coefficients = {
            "intercept": float(model.intercept_),
            "b_d": float(model.coef_[0]),
            "b_w": float(model.coef_[1]),
            "b_m": float(model.coef_[2]),
        }
        self._model = model
        return self

    def forecast_log_variance(self, rv: pd.Series) -> float:
        """One-step-ahead forecast of log(RV_{t+1}) given history `rv`."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() first.")
        features = self._build_features(rv).iloc[-1:][["log_rv_d", "log_rv_w", "log_rv_m"]].values
        return float(self._model.predict(features)[0])

    def forecast_variance(self, rv: pd.Series) -> float:
        """One-step-ahead forecast of RV_{t+1} (back-transformed)."""
        return float(np.exp(self.forecast_log_variance(rv)))
