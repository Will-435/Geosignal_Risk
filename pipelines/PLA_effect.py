"""Event study: TSMC price reaction around major PLA exercise dates."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT_PARQUET = PROJECT_ROOT / 'data' / 'processed' / 'pla_event_study_bp.parquet'
OUT_PNG = PROJECT_ROOT / 'data' / 'visualisations' / 'pla_event_study_bp.png'

DEFAULT_TICKER = os.environ.get('PLA_TICKER', 'TSM')

BASIS_POINTS_PER_UNIT = 10000.0
BUFFER_DAYS = 60
PLOT_DPI = 180

EVENT_WINDOWS = [
    (0, 1),
    (-1, 1),
    (-2, 2),
    (-5, 5),
]

PLA_EVENTS = [
    {'name': 'War games (Pelosi aftermath)', 'start': '2022-08-04', 'end': '2022-08-10'},
    {'name': 'Joint Sword (2023)',           'start': '2023-04-08', 'end': '2023-04-10'},
    {'name': 'Joint Sword-2024A',            'start': '2024-05-23', 'end': '2024-05-24'},
    {'name': 'Joint Sword-2024B',            'start': '2024-10-14', 'end': '2024-10-15'},
    {'name': 'Strait Thunder-2025A',         'start': '2025-04-01', 'end': '2025-04-02'},
    {'name': 'Justice Mission 2025',         'start': '2025-12-29', 'end': '2025-12-31'},
]


def to_basis_points(value):
    """Convert a fractional return to basis points."""
    return BASIS_POINTS_PER_UNIT * float(value)


def get_price_history(ticker, start, end):
    """Pull a clean OHLC dataframe from yfinance for the given ticker and window."""
    df = yf.download(ticker, start = start, end = end, progress = False, auto_adjust = False)

    if df is None or df.empty:
        raise ValueError(f"no data returned for {ticker}, check symbol and connectivity")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df = df.rename(columns = {'Datetime': 'Date'})
    if 'Date' not in df.columns:
        raise ValueError(f"expected Date column, got: {list(df.columns)}")

    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce').dt.date
    needed = ['Open', 'Close']
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")

    df['Open'] = pd.to_numeric(df['Open'], errors = 'coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors = 'coerce')

    return df.dropna(subset = ['Date', 'Open', 'Close']).reset_index(drop = True)


def nearest_trading_index(dates, target_date):
    """Return the first index where dates[i] >= target_date."""
    idx = np.searchsorted(dates, target_date)
    if idx >= len(dates):
        return len(dates) - 1
    return int(idx)


def compute_event_metrics(prices, event):
    """Compute reopen-gap and cumulative bp metrics across each window for one event."""
    dates = prices['Date'].values
    start_date = pd.to_datetime(event['start']).date()
    event_open_idx = nearest_trading_index(dates, start_date)

    event_start_trading_date = prices.loc[event_open_idx, 'Date']

    if event_open_idx == 0:
        reopen_gap_bp = np.nan
    else:
        open_today = float(prices.loc[event_open_idx, 'Open'])
        close_yesterday = float(prices.loc[event_open_idx - 1, 'Close'])
        reopen_gap_bp = to_basis_points(open_today / close_yesterday - 1.0)

    metrics = {
        'event': str(event['name']),
        'start': str(event['start']),
        'end': str(event['end']),
        'event_start_trading_date': str(event_start_trading_date),
        'reopen_gap_bp': float(reopen_gap_bp) if np.isfinite(reopen_gap_bp) else np.nan,
    }

    closes = prices['Close'].to_numpy(dtype = float)
    for (left, right) in EVENT_WINDOWS:
        left_idx = max(0, event_open_idx + int(left))
        right_idx = min(len(prices) - 1, event_open_idx + int(right))

        if right_idx <= left_idx:
            metrics[f'cc_bp_{left}_{right}'] = np.nan
            metrics[f'max_dd_bp_{left}_{right}'] = np.nan
            continue

        cumulative_return = float(closes[right_idx] / closes[left_idx] - 1.0)
        metrics[f'cc_bp_{left}_{right}'] = to_basis_points(cumulative_return)

        window_closes = closes[left_idx:right_idx + 1].astype(float)
        peak = np.maximum.accumulate(window_closes)
        drawdown = (window_closes / peak) - 1.0
        metrics[f'max_dd_bp_{left}_{right}'] = to_basis_points(float(drawdown.min()))

    return metrics


def main():
    """Pull TSMC prices over the event range, compute per-event metrics, plot and save."""
    OUT_PARQUET.parent.mkdir(parents = True, exist_ok = True)
    OUT_PNG.parent.mkdir(parents = True, exist_ok = True)

    earliest_start = min(pd.to_datetime(event['start']) for event in PLA_EVENTS) - pd.Timedelta(days = BUFFER_DAYS)
    latest_end = max(pd.to_datetime(event['end']) for event in PLA_EVENTS) + pd.Timedelta(days = BUFFER_DAYS)

    prices = get_price_history(DEFAULT_TICKER, str(earliest_start.date()), str(latest_end.date()))

    rows = [compute_event_metrics(prices, event) for event in PLA_EVENTS]
    df = pd.DataFrame(rows).sort_values('start').reset_index(drop = True)

    for col in df.columns:
        is_non_scalar = df[col].apply(
            lambda value: isinstance(value, (list, tuple, dict, pd.Series, np.ndarray))
        )
        if is_non_scalar.any():
            raise ValueError(f"non-scalar values found in column: {col}")

    df.to_parquet(OUT_PARQUET, index = False)

    bp_values = pd.to_numeric(df['cc_bp_-1_1'], errors = 'coerce').to_numpy(dtype = float)
    finite_mask = np.isfinite(bp_values)
    x_positions = np.arange(len(bp_values))[finite_mask]
    y_values = bp_values[finite_mask]

    plt.figure(figsize = (12, 5))
    plt.bar(x_positions, y_values)
    plt.axhline(0, linestyle = '--', linewidth = 1)
    plt.xticks(x_positions, df.loc[finite_mask, 'start'].tolist(), rotation = 45, ha = 'right')
    plt.title(f'{DEFAULT_TICKER} bp move around PLA exercise start (Close-Close, [-1,+1])')
    plt.xlabel('Event start date')
    plt.ylabel('bp')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi = PLOT_DPI)
    plt.close()

    print(f"event study saved: {len(df)} events")
    print(df[['event', 'start', 'reopen_gap_bp', 'cc_bp_-1_1', 'cc_bp_-2_2', 'max_dd_bp_-5_5']])


if __name__ == '__main__':
    main()
