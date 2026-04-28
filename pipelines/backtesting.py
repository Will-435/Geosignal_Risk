"""Backtest the volatility-regime de-risking strategy against buy-and-hold."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'signal_generator' / 'rf_signal_model.pkl'

OUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'backtest_vol_strategy.csv'
PLOTS_DIR = PROJECT_ROOT / 'data' / 'visualisations'

EQUITY_PNG = PLOTS_DIR / 'backtest_vol_strategy.png'
PROB_AND_EXPOSURE_PNG = PLOTS_DIR / 'backtest_vol_strategy_p_and_exposure.png'
DRAWDOWN_PNG = PLOTS_DIR / 'backtest_vol_strategy_drawdown.png'
RETURN_HIST_PNG = PLOTS_DIR / 'backtest_vol_strategy_return_hist.png'

DEFAULT_SYMBOLS = ['TSM', 'TSMC', '2330.TW', 'TSM.N', 'TSM.US']
NON_FEATURE_COLS = {'date', 'symbol', 'adj_close'}

DEFAULT_RISK_OFF_THRESH = 0.60
DEFAULT_RISK_OFF_QUANTILE = 0.80
DEFAULT_EXPOSURE_FLOOR = 0.10
MIN_FEATURE_OVERLAP_FRACTION = 0.10
MIN_FEATURE_OVERLAP_ABSOLUTE = 5
HISTOGRAM_BINS = 60
SAMPLE_SYMBOLS_FOR_ERROR = 25
PLOT_DPI = 180

env_symbol = os.environ.get('BT_SYMBOL', '').strip().upper()
SYMBOL_OVERRIDE = env_symbol if env_symbol else None

RISK_OFF_THRESH = float(os.environ.get('RISK_OFF_THRESH', str(DEFAULT_RISK_OFF_THRESH)))
RISK_OFF_QUANTILE = float(os.environ.get('RISK_OFF_Q', str(DEFAULT_RISK_OFF_QUANTILE)))
USE_DYNAMIC_THRESH = int(os.environ.get('USE_DYNAMIC_THRESH', '1'))
USE_SMOOTH_EXPOSURE = int(os.environ.get('USE_SMOOTH_EXPOSURE', '1'))
EXPOSURE_FLOOR = float(os.environ.get('EXPOSURE_FLOOR', str(DEFAULT_EXPOSURE_FLOOR)))


def numeric_feature_columns(df):
    """Return all numeric columns of df that are not date/symbol/adj_close."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def compute_drawdown(equity):
    """Return the drawdown series given an equity curve."""
    peak = equity.cummax()
    return (equity / peak) - 1.0


def pick_symbol(df):
    """Choose the symbol to backtest, honouring BT_SYMBOL override if available."""
    symbols = set(df['symbol'].unique())
    if SYMBOL_OVERRIDE is not None:
        return SYMBOL_OVERRIDE
    chosen = next((s for s in DEFAULT_SYMBOLS if s in symbols), None)
    if chosen is None:
        sample = sorted(list(symbols))[:SAMPLE_SYMBOLS_FOR_ERROR]
        raise ValueError(f"no default TSM symbol found, available sample: {sample}")
    return chosen


def align_features_to_model(model, feature_frame):
    """Reindex the feature frame to match the columns the trained model expects."""
    if not hasattr(model, 'feature_names_in_'):
        return feature_frame, None

    expected = list(model.feature_names_in_)
    present = sorted(set(expected) & set(feature_frame.columns))
    minimum_overlap = max(MIN_FEATURE_OVERLAP_ABSOLUTE,
                          int(MIN_FEATURE_OVERLAP_FRACTION * len(expected)))

    if len(present) < minimum_overlap:
        raise ValueError(
            f"feature mismatch too large: model expects {len(expected)} cols, "
            f"only {len(present)} present in merged_features.csv. retrain or rebuild features."
        )
    return feature_frame.reindex(columns=expected, fill_value=0.0), expected


def score_probabilities(model, feature_matrix):
    """Return p_high_vol for the supplied feature matrix, falling back to predict()."""
    try:
        return np.clip(model.predict_proba(feature_matrix)[:, 1], 0.0, 1.0)
    except Exception:
        predictions = model.predict(feature_matrix)
        return (predictions == 1).astype(float)


def build_exposure(daily, smooth):
    """Compute the next-day exposure series (smooth = inverse of p_high_vol)."""
    if smooth:
        raw = 1.0 - daily['p_high_vol'].to_numpy(dtype=float)
        clipped = np.clip(raw, EXPOSURE_FLOOR, 1.0)
        return pd.Series(clipped, index=daily.index).shift(1).fillna(1.0)
    return (1 - daily['risk_off']).shift(1).fillna(1.0)


def save_equity_plot(daily, symbol):
    """Plot strategy vs buy-and-hold equity curves."""
    plt.figure(figsize=(12, 5))
    plt.plot(daily['date'], daily['buy_hold_equity'], label='Buy & Hold')
    plt.plot(daily['date'], daily['strategy_equity'], label='De-risk on HIGH_VOL')
    plt.title(f'Volatility-regime backtest ({symbol})')
    plt.xlabel('Date')
    plt.ylabel('Equity (normalised)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_PNG, dpi=PLOT_DPI)
    plt.close()


def save_prob_exposure_plot(daily, symbol, threshold):
    """Plot p_high_vol against the exposure schedule it implies."""
    plt.figure(figsize=(12, 5))
    plt.plot(daily['date'], daily['p_high_vol'], label='p_high_vol')
    plt.axhline(threshold, linestyle='--', linewidth=1, label='risk_off_threshold')
    plt.plot(daily['date'], daily['exposure_next'], label='exposure_next')
    plt.title(f'p_high_vol and exposure over time ({symbol})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PROB_AND_EXPOSURE_PNG, dpi=PLOT_DPI)
    plt.close()


def save_drawdown_plot(daily, symbol):
    """Plot drawdown comparison."""
    plt.figure(figsize=(12, 5))
    plt.plot(daily['date'], daily['buy_hold_dd'], label='Buy & Hold drawdown')
    plt.plot(daily['date'], daily['strategy_dd'], label='De-risk drawdown')
    plt.title(f'Drawdown comparison ({symbol})')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DRAWDOWN_PNG, dpi=PLOT_DPI)
    plt.close()


def save_return_histogram(daily, symbol):
    """Plot next-day return distributions split by prior-day risk regime."""
    tmp = daily.copy()
    tmp['risk_off_prev'] = tmp['risk_off'].shift(1)

    risk_on_returns = tmp.loc[tmp['risk_off_prev'] == 0, 'ret_1d'].dropna().to_numpy()
    risk_off_returns = tmp.loc[tmp['risk_off_prev'] == 1, 'ret_1d'].dropna().to_numpy()

    plt.figure(figsize=(12, 5))
    plotted_any = False
    if risk_on_returns.size > 0:
        plt.hist(risk_on_returns, bins=HISTOGRAM_BINS, alpha=0.6,
                 label='Next-day ret | prior = RISK_ON')
        plotted_any = True
    if risk_off_returns.size > 0:
        plt.hist(risk_off_returns, bins=HISTOGRAM_BINS, alpha=0.6,
                 label='Next-day ret | prior = RISK_OFF')
        plotted_any = True

    plt.title(f'Next-day returns by regime ({symbol})')
    if plotted_any:
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'no data', ha='center', va='center')
    plt.xlabel('Daily return')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RETURN_HIST_PNG, dpi=PLOT_DPI)
    plt.close()


def main():
    """Score the model day-by-day, build equity curves, save plots and CSV."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"missing: {FEATURES_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"missing: {MODEL_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    if 'date' not in df.columns or 'symbol' not in df.columns:
        raise ValueError("features file must include date and symbol columns")
    if 'adj_close' not in df.columns:
        raise ValueError("features file must include adj_close for backtesting")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()

    symbol_to_use = pick_symbol(df)
    df = df[df['symbol'] == symbol_to_use].copy()
    if df.empty:
        raise ValueError(f"no rows for symbol {symbol_to_use}")
    df = df.sort_values('date').reset_index(drop=True)

    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("no numeric feature columns found")

    daily = df.groupby('date', as_index=False)[['adj_close'] + feature_cols].mean()
    daily = daily.sort_values('date').reset_index(drop=True)

    feature_matrix = daily[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_matrix, _ = align_features_to_model(model, feature_matrix)

    daily['p_high_vol'] = score_probabilities(model, feature_matrix)

    threshold = RISK_OFF_THRESH
    if USE_DYNAMIC_THRESH == 1:
        threshold = float(np.nanquantile(daily['p_high_vol'], RISK_OFF_QUANTILE))
    daily['risk_off'] = (daily['p_high_vol'] >= threshold).astype(int)

    daily['exposure_next'] = build_exposure(daily, smooth=(USE_SMOOTH_EXPOSURE == 1))

    daily['ret_1d'] = daily['adj_close'].pct_change(fill_method=None)
    daily['strategy_ret'] = daily['exposure_next'] * daily['ret_1d']
    daily['buy_hold_ret'] = daily['ret_1d']

    daily['strategy_equity'] = (1 + daily['strategy_ret'].fillna(0.0)).cumprod()
    daily['buy_hold_equity'] = (1 + daily['buy_hold_ret'].fillna(0.0)).cumprod()
    daily['strategy_dd'] = compute_drawdown(daily['strategy_equity'])
    daily['buy_hold_dd'] = compute_drawdown(daily['buy_hold_equity'])

    daily.to_csv(OUT_CSV, index=False)

    save_equity_plot(daily, symbol_to_use)
    save_prob_exposure_plot(daily, symbol_to_use, threshold)
    save_drawdown_plot(daily, symbol_to_use)
    save_return_histogram(daily, symbol_to_use)

    risk_off_pct = 100.0 * daily['risk_off'].mean()
    print(f"symbol: {symbol_to_use}")
    print(f"risk-off freq: {risk_off_pct:.1f}%")
    print(f"threshold: {threshold:.4f}")
    print(f"final strategy equity: {daily['strategy_equity'].iloc[-1]:.3f}")
    print(f"final buy-hold equity: {daily['buy_hold_equity'].iloc[-1]:.3f}")


if __name__ == '__main__':
    main()
