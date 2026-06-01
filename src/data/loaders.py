"""NIFTY 50 data loaders.

Two data sources supported:

1. **yfinance** — convenient for daily OHLCV from Yahoo Finance (free, no auth).
   Symbol: `^NSEI` for NIFTY 50.

2. **NSE bhavcopy archives** — direct daily CSV downloads from NSE.
   Useful for accessing 20+ years of history, individual stocks, and richer fields.

For volatility forecasting research, daily OHLCV from yfinance is sufficient to
get started. Switch to bhavcopy when you need intraday or longer history.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def download_nifty_yfinance(
    start: str = "2007-01-01",
    end: Optional[str] = None,
    symbol: str = "^NSEI",
    save_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Download daily OHLCV for NIFTY 50 via yfinance.

    Parameters
    ----------
    start, end : str
        Date range in YYYY-MM-DD format. `end=None` means up to today.
    symbol : str
        Yahoo Finance symbol. `^NSEI` is NIFTY 50.
    save_path : Path | str, optional
        If provided, save the DataFrame to this path as parquet.

    Returns
    -------
    pd.DataFrame with columns Open, High, Low, Close, Volume, indexed by Date.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is required. `pip install yfinance`.") from e

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbol}")

    # Flatten multi-index columns (yfinance returns Ticker level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Adj Close": "AdjClose"})
    df.index.name = "Date"

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)

    return df


def load_nifty(path: Path | str) -> pd.DataFrame:
    """Load previously saved NIFTY OHLCV from parquet."""
    return pd.read_parquet(path)


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns from a Close price series."""
    import numpy as np
    return np.log(close / close.shift(1)).rename("log_return")


def compute_squared_returns(close: pd.Series) -> pd.Series:
    """Squared log returns (a noisy proxy for daily realized variance)."""
    ret = compute_log_returns(close)
    return (ret ** 2).rename("sq_return")
