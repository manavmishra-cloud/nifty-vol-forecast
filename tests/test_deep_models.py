"""Smoke tests for LSTM and Transformer volatility forecasters.

These verify the models load, train for a few epochs, and produce sensible
predictions on synthetic data. They don't validate forecast quality — that's
the job of the walk-forward evaluation pipeline.
"""
import numpy as np
import pandas as pd
import pytest

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

if torch_available:
    from src.models.deep_models import (
        LSTMForecaster,
        TransformerForecaster,
        TrainConfig,
        build_sequences,
        train_deep_model,
        predict_deep_model,
    )


@pytest.fixture
def synthetic_volatility_panel():
    """40 days of synthetic volatility-like data with autocorrelated log RV."""
    rng = np.random.default_rng(0)
    n = 40
    log_rv = -8.0 + 0.6 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    df = pd.DataFrame({
        "log_return": rng.standard_normal(n) * 0.01,
        "log_rv": log_rv,
        "rv_p": np.exp(log_rv + 0.1 * rng.standard_normal(n)),
        "rv_gk": np.exp(log_rv + 0.1 * rng.standard_normal(n)),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
def test_build_sequences_shape(synthetic_volatility_panel):
    df = synthetic_volatility_panel
    X, y, dates = build_sequences(df, ["log_return", "log_rv"], "log_rv", seq_len=5)
    assert X.shape == (len(df) - 5, 5, 2)
    assert y.shape == (len(df) - 5,)
    assert len(dates) == len(y)


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
def test_lstm_train_and_predict(synthetic_volatility_panel):
    df = synthetic_volatility_panel
    X, y, _ = build_sequences(df, ["log_return", "log_rv", "rv_p", "rv_gk"], "log_rv", seq_len=5)

    model = LSTMForecaster(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    fitted = train_deep_model(
        model, X[:25], y[:25],
        config=TrainConfig(epochs=5, batch_size=4, verbose=False),
    )
    preds = predict_deep_model(fitted, X[25:30])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
def test_transformer_train_and_predict(synthetic_volatility_panel):
    df = synthetic_volatility_panel
    X, y, _ = build_sequences(df, ["log_return", "log_rv", "rv_p", "rv_gk"], "log_rv", seq_len=5)

    model = TransformerForecaster(
        n_features=4, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, dropout=0.0, seq_len=5,
    )
    fitted = train_deep_model(
        model, X[:25], y[:25],
        config=TrainConfig(epochs=5, batch_size=4, verbose=False),
    )
    preds = predict_deep_model(fitted, X[25:30])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


@pytest.mark.skipif(not torch_available, reason="PyTorch not available")
def test_standardization_preserves_scale(synthetic_volatility_panel):
    """Predictions should be in the same scale as the input target, not in z-scored units."""
    df = synthetic_volatility_panel
    X, y, _ = build_sequences(df, ["log_return", "log_rv"], "log_rv", seq_len=5)

    model = LSTMForecaster(n_features=2, hidden_size=8, num_layers=1, dropout=0.0)
    fitted = train_deep_model(
        model, X[:25], y[:25],
        config=TrainConfig(epochs=20, batch_size=4, verbose=False),
    )
    preds = predict_deep_model(fitted, X[25:30])

    # Predictions should be on roughly the same order of magnitude as y (~-8 for log RV)
    assert -10 < preds.mean() < -6
