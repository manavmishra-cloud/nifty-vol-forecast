"""Deep sequence models for volatility forecasting.

Two architectures implemented:

1. **LSTM** — Standard 2-layer LSTM with dropout. Reads a sequence of past daily
   features (log returns, log realized variance, range estimators) and predicts
   the next day's log realized variance.

2. **Transformer (encoder-only)** — Self-attention over the same sequence,
   with sinusoidal positional encoding. Functionally similar to LSTM but
   captures long-range dependencies via attention rather than recurrence.

Both models share the same input/output contract:
    Input:  (batch, seq_len, n_features) — past daily features
    Output: (batch, 1) — predicted log realized variance for next day

Training is done with early stopping on a validation split. Walk-forward
evaluation retrains every `refit_freq` days (default ~250 ≈ once per year)
to balance compute cost against staleness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sequence dataset construction
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = 22,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y, idx) arrays of overlapping sequences from a time-indexed DataFrame.

    For row t, X[t] contains features from rows [t - seq_len + 1, t] and y[t]
    is the target at row t + 1 (so we're predicting one step ahead).

    Returns
    -------
    X : (n_samples, seq_len, n_features)
    y : (n_samples,)
    idx : DatetimeIndex of length n_samples, giving the *prediction* date
          (the row corresponding to y).
    """
    features = df[feature_cols].values
    targets = df[target_col].values

    n = len(df)
    n_samples = n - seq_len  # need seq_len history + 1 step ahead

    X = np.empty((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)

    for t in range(n_samples):
        X[t] = features[t : t + seq_len]
        y[t] = targets[t + seq_len]  # predict the row just after the window

    pred_dates = df.index[seq_len:]  # length n_samples
    return X, y, pred_dates


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMForecaster(nn.Module):
        """2-layer LSTM with dropout, predicting a single scalar."""

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F)
            out, _ = self.lstm(x)
            # Use last timestep's hidden state
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


    class PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding from Vaswani et al. 2017."""

        def __init__(self, d_model: int, max_len: int = 100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, d_model)
            return x + self.pe[:, : x.size(1), :]


    class TransformerForecaster(nn.Module):
        """Encoder-only Transformer with positional encoding."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
            seq_len: int = 22,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F)
            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h)  # (B, T, d_model)
            # Use last timestep
            last = h[:, -1, :]
            return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Training & inference helpers
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    early_stopping_patience: int = 10
    device: str = "cpu"
    verbose: bool = False


@dataclass
class FittedDeepModel:
    """Wraps a trained PyTorch model with the scaler statistics used at fit time."""
    model: "nn.Module"
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    seq_len: int
    config: TrainConfig


def _standardize(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z-score normalize features per dimension; centre target."""
    # Reshape X to compute per-feature stats across (samples, time)
    n, T, F = X.shape
    flat = X.reshape(-1, F)
    feature_mean = flat.mean(axis=0)
    feature_std = flat.std(axis=0) + 1e-8
    X_n = (X - feature_mean) / feature_std

    target_mean = float(y.mean())
    target_std = float(y.std() + 1e-8)
    y_n = (y - target_mean) / target_std

    return X_n, y_n, {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def train_deep_model(
    model: "nn.Module",
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig | None = None,
) -> FittedDeepModel:
    """Train a deep model with early stopping on the tail of the training set as val.

    Walk-forward semantics: caller has already sliced X/y to the training window.
    We do a temporal val split: last `val_frac` of samples (still chronological).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. `pip install torch`.")

    config = config or TrainConfig()
    X_n, y_n, scaler = _standardize(X, y)

    # Temporal split — val is the tail
    n_val = max(int(len(X_n) * config.val_frac), 1)
    X_train, y_train = X_n[:-n_val], y_n[:-n_val]
    X_val, y_val = X_n[-n_val:], y_n[-n_val:]

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_left = config.early_stopping_patience

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        if config.verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}: train={np.mean(train_losses):.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if config.verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FittedDeepModel(
        model=model,
        feature_mean=scaler["feature_mean"],
        feature_std=scaler["feature_std"],
        target_mean=scaler["target_mean"],
        target_std=scaler["target_std"],
        seq_len=X.shape[1],
        config=config,
    )


def predict_deep_model(fitted: FittedDeepModel, X: np.ndarray) -> np.ndarray:
    """Inference: applies the same scaling, returns predictions in original units."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    X_n = (X - fitted.feature_mean) / fitted.feature_std
    fitted.model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_n).float().to(fitted.config.device)
        pred_n = fitted.model(x_t).cpu().numpy()
    # Un-standardize
    return pred_n * fitted.target_std + fitted.target_mean
