# Data — NIFTY 50

## Source: yfinance (default, free, no auth required)

The training pipeline auto-downloads daily OHLCV via yfinance:

```bash
python -c "from src.data.loaders import download_nifty_yfinance; \
  download_nifty_yfinance(start='2007-01-01', save_path='data/processed/nifty.parquet')"
```

Or pass `--download` to `train_all.py` which will fetch on first run.

## Source: NSE bhavcopy (for production, longer history, individual stocks)

NSE publishes daily bhavcopy files. They're CSV downloads from
`https://www.nseindia.com/products/content/equities/equities/eq_security.htm`
— may require manual scraping or the `nsepython` / `nselib` packages.

Add a bhavcopy downloader to `src/data/loaders.py` when you need it.

## What's gitignored

`data/raw/` and `data/processed/` are in `.gitignore`. Each developer downloads
or generates their own data.
