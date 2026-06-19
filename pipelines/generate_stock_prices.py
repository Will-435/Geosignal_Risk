import os
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'stock_prices.parquet'

LOOKBACK_YEARS = 3
TICKERS = {
    "TSMC": "TSM",
    "INTC": "INTC",
    "SMSN": "005930.KS",
}
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def get_date_range():
    """Return start and end dates for the price pull, configurable via env vars."""
    end_env = os.environ.get("STOCK_END_DATE")
    start_env = os.environ.get("STOCK_START_DATE")
    end = pd.to_datetime(end_env).date() if end_env else datetime.now(UTC).date()
    start = (
        pd.to_datetime(start_env).date() if start_env
        else end - timedelta(days = 365 * LOOKBACK_YEARS)
    )
    return start, end


def fetch_prices():
    """Pull daily OHLCV bars for each configured ticker from yfinance."""
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is required, run pip install yfinance") from exc

    start_date, end_date = get_date_range()

    raw = yf.download(
        list(TICKERS.values()),
        start = start_date.isoformat(),
        end = (end_date + timedelta(days = 1)).isoformat(),
        auto_adjust = False,
        progress = False,
        group_by = "column",
        threads = True,
    )
    if raw.empty:
        raise RuntimeError("no price data returned, check tickers or connectivity")

    frames = []
    for label, yahoo_ticker in TICKERS.items():
        per_symbol = pd.DataFrame({"date": raw.index})
        for col in PRICE_COLUMNS:
            key = (col, yahoo_ticker) if isinstance(raw.columns, pd.MultiIndex) else col
            target_col = col.lower().replace(" ", "_")
            per_symbol[target_col] = raw[key].values if key in raw.columns else pd.NA
        per_symbol["symbol"] = label
        frames.append(per_symbol)

    combined = pd.concat(frames, ignore_index = True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date.astype(str)

    ordered_cols = ["date", "symbol", "adj_close", "close", "open", "high", "low", "volume"]
    return combined[[column for column in ordered_cols if column in combined.columns]]


if __name__ == "__main__":
    df = fetch_prices()
    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(OUTPUT_PATH, index = False)
    print(f"saved {len(df)} rows of stock prices")
