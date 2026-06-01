"""Machine-learning baselines for volatility forecasting.

Compared to GARCH/HAR-RV, these are flexible non-linear models that can
exploit a richer feature panel — lagged RVs, returns, range estimators,
volume features, calendar effects.

Implemented:

- **XGBoost** — standard strong baseline for tabular time-series.
- **LightGBM** — alternative tree boosting (often faster, similar accuracy).

Forecast target: log realized variance at horizon t+1 (regression task).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class XGBoostVolForecaster:
    """XGBoost regression on engineered volatility features.

    Target = log(RV_{t+1}). Predict and exponentiate to get variance forecast.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost required. `pip install xgboost`.") from e
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            tree_method="hist",
            objective="reg:squarederror",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostVolForecaster":
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


class LightGBMVolForecaster:
    """LightGBM regression — alternative to XGBoost."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm required. `pip install lightgbm`.") from e
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMVolForecaster":
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)


def build_volatility_features(
    df: pd.DataFrame,
    log_rv_col: str = "log_rv_yz",
    return_col: str = "log_return",
    lags: tuple[int, ...] = (1, 2, 3, 5, 10, 22),
) -> pd.DataFrame:
    """Build the input feature panel for ML vol forecasting.

    Includes:
    - Lagged log realized variance (multiple lags)
    - Lagged squared returns
    - Rolling means/std of returns (5, 22 day)
    - Lagged absolute returns
    - Day-of-week, month-of-year dummies
    """
    features = pd.DataFrame(index=df.index)

    log_rv = np.log(df[log_rv_col.replace("log_", "")].replace(0, np.nan))

    # Lagged log RV
    for lag in lags:
        features[f"log_rv_lag{lag}"] = log_rv.shift(lag)

    # HAR-style aggregates
    features["log_rv_w5"] = log_rv.shift(1).rolling(5).mean()
    features["log_rv_m22"] = log_rv.shift(1).rolling(22).mean()

    # Lagged returns
    ret = df[return_col]
    features["ret_lag1"] = ret.shift(1)
    features["ret_lag2"] = ret.shift(2)
    features["abs_ret_lag1"] = ret.shift(1).abs()
    features["sq_ret_lag1"] = ret.shift(1) ** 2

    # Rolling return statistics
    features["ret_mean_w5"] = ret.shift(1).rolling(5).mean()
    features["ret_std_w5"] = ret.shift(1).rolling(5).std()
    features["ret_std_w22"] = ret.shift(1).rolling(22).std()

    # Calendar effects
    features["dow"] = df.index.dayofweek
    features["month"] = df.index.month

    return features
