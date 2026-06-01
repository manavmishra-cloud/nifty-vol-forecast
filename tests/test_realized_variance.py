"""Smoke tests for realized variance estimators."""
import numpy as np
import pandas as pd
import pytest

from src.data.realized_variance import (
    parkinson_variance,
    garman_klass_variance,
    rogers_satchell_variance,
    yang_zhang_variance,
    add_realized_variance_columns,
)


@pytest.fixture
def synthetic_ohlcv():
    """5 days of synthetic OHLCV data with deliberate volatility variation."""
    return pd.DataFrame({
        "Open":   [100.0, 101.5, 99.0, 102.0, 100.5],
        "High":   [102.0, 103.0, 100.5, 103.0, 101.5],
        "Low":    [99.5, 100.0, 98.0, 101.0, 99.5],
        "Close":  [101.5, 100.5, 99.5, 102.5, 100.0],
        "Volume": [1e6, 1.2e6, 1.5e6, 1.1e6, 1.3e6],
    }, index=pd.date_range("2024-01-01", periods=5, freq="B"))


def test_parkinson_positive(synthetic_ohlcv):
    rv = parkinson_variance(synthetic_ohlcv["High"], synthetic_ohlcv["Low"])
    assert (rv >= 0).all()
    assert not rv.isna().any()


def test_garman_klass_positive(synthetic_ohlcv):
    rv = garman_klass_variance(
        synthetic_ohlcv["Open"], synthetic_ohlcv["High"],
        synthetic_ohlcv["Low"], synthetic_ohlcv["Close"],
    )
    assert (rv >= 0).all()


def test_rogers_satchell_positive(synthetic_ohlcv):
    rv = rogers_satchell_variance(
        synthetic_ohlcv["Open"], synthetic_ohlcv["High"],
        synthetic_ohlcv["Low"], synthetic_ohlcv["Close"],
    )
    assert (rv >= 0).all()


def test_yang_zhang_first_nan(synthetic_ohlcv):
    rv = yang_zhang_variance(
        synthetic_ohlcv["Open"], synthetic_ohlcv["High"],
        synthetic_ohlcv["Low"], synthetic_ohlcv["Close"],
    )
    # First value depends on close.shift(1) which is NaN
    assert pd.isna(rv.iloc[0])
    # Subsequent values should be positive
    assert (rv.iloc[1:] >= 0).all()


def test_add_realized_variance_columns(synthetic_ohlcv):
    df = add_realized_variance_columns(synthetic_ohlcv)
    expected_cols = ["log_return", "sq_return", "rv_parkinson", "rv_garman_klass",
                     "rv_rogers_satchell", "rv_yang_zhang", "rv_yz_annualized"]
    for col in expected_cols:
        assert col in df.columns
