"""Compute per-symbol market features and the volatility-regime target label."""

import os
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'merged_features.parquet'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.parquet'

FORWARD_VOL_WINDOW = int(os.environ.get('FWD_VOL_WINDOW', '5'))
QUANTILE_LOOKBACK = int(os.environ.get('VOL_Q_LOOKBACK', '252'))
QUANTILE_LEVEL = float(os.environ.get('VOL_Q_LEVEL', '0.75'))
QUANTILE_MIN_PERIODS = int(os.environ.get('VOL_Q_MIN_PERIODS', '60'))

SHORT_RETURN_LAG = 1
LONG_RETURN_LAG = 5
MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
DAILY_VOL_WINDOW = 10
DAILY_VOL_MIN_PERIODS = 2

REQUIRED_COLUMNS = ['date', 'symbol', 'adj_close']


def require_cols(df, cols):
    """Raise if any of the named columns are missing from the dataframe."""
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f'merged_features.parquet missing required columns: {missing}')


def add_market_features(df):
    """Add lagged returns, moving averages, and rolling daily volatility per symbol."""
    df = df.copy().sort_values(['symbol', 'date'], kind = 'mergesort')

    df['ret_1d'] = df.groupby('symbol')['adj_close'].pct_change(SHORT_RETURN_LAG, fill_method = None)
    df['ret_5d'] = df.groupby('symbol')['adj_close'].pct_change(LONG_RETURN_LAG, fill_method = None)

    df['ma_5d'] = df.groupby('symbol')['adj_close'].transform(
        lambda series: series.rolling(MA_SHORT_WINDOW, min_periods = 1).mean()
    )
    df['ma_20d'] = df.groupby('symbol')['adj_close'].transform(
        lambda series: series.rolling(MA_LONG_WINDOW, min_periods = 1).mean()
    )

    daily_returns = df.groupby('symbol')['adj_close'].pct_change(fill_method = None)
    df['vol_10d'] = daily_returns.groupby(df['symbol']).transform(
        lambda series: series.rolling(DAILY_VOL_WINDOW, min_periods = DAILY_VOL_MIN_PERIODS).std()
    )

    return df


def add_forward_vol_and_threshold(df):
    """
    Add forward 5 day realised volatility and its rolling 75th percentile
    threshold. The threshold uses past values only, shifted by one day, to
    prevent look ahead leakage.

    INPUTS:
        * df, dataframe with date, symbol and adj_close columns

    OUTPUTS:
        * the dataframe with two new columns, vol_fwd_5d and vol_q75_252d
    """
    df = df.copy().sort_values(['symbol', 'date'], kind = 'mergesort')

    daily_returns = df.groupby('symbol')['adj_close'].pct_change(fill_method = None)
    forward_base = daily_returns.groupby(df['symbol']).shift(-1)

    forward_vol = (
        forward_base.groupby(df['symbol'])
        .transform(lambda series: series.rolling(FORWARD_VOL_WINDOW, min_periods = FORWARD_VOL_WINDOW).std())
        .groupby(df['symbol']).shift(-(FORWARD_VOL_WINDOW - 1))
    )
    df['vol_fwd_5d'] = forward_vol

    df['vol_q75_252d'] = (
        df.groupby('symbol')['vol_fwd_5d']
        .transform(lambda series: series.rolling(
            QUANTILE_LOOKBACK, min_periods = QUANTILE_MIN_PERIODS
        ).quantile(QUANTILE_LEVEL))
        .groupby(df['symbol']).shift(1)
    )

    return df


def ensure_target_label(df):
    """Build the HIGH_VOL/LOW_VOL target column if it does not already exist."""
    if 'target_label' in df.columns:
        if 'vol_fwd_5d' not in df.columns or 'vol_q75_252d' not in df.columns:
            df = add_forward_vol_and_threshold(df)
        return df

    df = add_forward_vol_and_threshold(df)

    target = pd.Series(index = df.index, dtype = 'object')
    valid = df['vol_fwd_5d'].notna() & df['vol_q75_252d'].notna()

    target[valid & (df['vol_fwd_5d'] > df['vol_q75_252d'])] = 'HIGH_VOL'
    target[valid & (df['vol_fwd_5d'] <= df['vol_q75_252d'])] = 'LOW_VOL'

    df['target_label'] = target
    return df


def main():
    """Read merged features, add market features and target label, save labelled output."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Merged feature file not found: {INPUT_PATH}')

    df = pd.read_parquet(INPUT_PATH)
    require_cols(df, REQUIRED_COLUMNS)
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')

    df = add_market_features(df)
    df = ensure_target_label(df)

    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(OUTPUT_PATH, index = False)
    print(f"labelled features saved: {len(df)} rows")


if __name__ == '__main__':
    main()
