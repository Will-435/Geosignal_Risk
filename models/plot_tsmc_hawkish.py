"""Plot TSMC's last 5 years of price with hawkish news days marked as dashed lines."""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRICES_PATH = PROJECT_ROOT / 'data' / 'raw' / 'stock_prices.parquet'
NEWS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'headlines_with_sentiment.parquet'
OUT_DIR = PROJECT_ROOT / 'data' / 'visualisations'
OUT_PATH = OUT_DIR / 'tsmc_price_with_hawkish_lines_5y.png'

TSM_CANDIDATES = ['TSM', 'TSMC', '2330.TW', 'TSM.N', 'TSM.US']
PRICE_COLUMN_CANDIDATES = ['adj_close', 'close', 'price']
LOOKBACK_YEARS = 5
PLOT_DPI = 180

DEFAULT_MIN_ARTICLES = 2
DEFAULT_TOP_HAWKISH_DAYS = 5
NEG_RANK_WEIGHT = 1.0
VADER_RANK_WEIGHT = 1.0

REQUIRED_NEWS_COLUMNS = {'date', 'vader_compound', 'finbert_sentiment'}


def find_tsm_symbol(prices):
    """Choose the best matching TSMC symbol from the prices file."""
    symbols = set(prices['symbol'].astype(str).unique())
    for candidate in TSM_CANDIDATES:
        if candidate in symbols:
            return candidate
    for symbol in symbols:
        if 'TSM' in symbol:
            return symbol
    raise ValueError(f"no TSMC symbol found, available sample: {list(symbols)[:20]}")


def aggregate_daily_news(news):
    """Aggregate per-day sentiment shares from a row-per-headline news frame."""
    news = news.copy()
    news['finbert_sentiment'] = news['finbert_sentiment'].astype(str).str.lower().str.strip()
    return news.groupby('date').agg(
        n_articles = ('finbert_sentiment', 'size'),
        vader_mean = ('vader_compound', 'mean'),
        neg_share = ('finbert_sentiment', lambda series: (series == 'negative').mean()),
        neu_share = ('finbert_sentiment', lambda series: (series == 'neutral').mean()),
        pos_share = ('finbert_sentiment', lambda series: (series == 'positive').mean()),
    ).reset_index()


def select_hawkish_days(daily, min_articles, top_n):
    """Rank by negativity and pick the top-N hawkish trading days."""
    filtered = daily[daily['n_articles'] >= min_articles].copy()
    if filtered.empty:
        print("no news days after MIN_ARTICLES filter, lower it to 1")
        return pd.Series([], dtype = 'datetime64[ns]')

    filtered['neg_rank'] = filtered['neg_share'].rank(pct = True, method = 'average')
    filtered['vader_rank'] = (-filtered['vader_mean']).rank(pct = True, method = 'average')
    filtered['hawkish_score'] = (
        NEG_RANK_WEIGHT * filtered['neg_rank']
        + VADER_RANK_WEIGHT * filtered['vader_rank']
    )

    top = filtered.sort_values(
        ['hawkish_score', 'neg_share', 'vader_mean', 'n_articles'],
        ascending = [False, False, True, False],
    ).head(top_n)

    print("top hawkish days (5Y):")
    print(top[['date', 'n_articles', 'neg_share', 'vader_mean', 'hawkish_score']].to_string(index = False))
    return top['date']


def main():
    """Plot TSMC price over the last 5 years with hawkish news days overlaid."""
    OUT_DIR.mkdir(parents = True, exist_ok = True)

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"missing prices file: {PRICES_PATH}")
    if not NEWS_PATH.exists():
        raise FileNotFoundError(f"missing news sentiment file: {NEWS_PATH}")

    prices = pd.read_parquet(PRICES_PATH).assign(date = lambda frame: pd.to_datetime(frame['date']))
    news = pd.read_parquet(NEWS_PATH).assign(date = lambda frame: pd.to_datetime(frame['date']))

    for col in ['date', 'symbol']:
        if col not in prices.columns:
            raise ValueError(f"prices file missing column: {col}")

    price_col = next((column for column in PRICE_COLUMN_CANDIDATES if column in prices.columns), None)
    if price_col is None:
        raise ValueError(f"no price column found in prices, tried: {PRICE_COLUMN_CANDIDATES}")

    if 'title' in news.columns:
        news['title'] = news['title'].fillna('')

    missing_news_cols = REQUIRED_NEWS_COLUMNS - set(news.columns)
    if missing_news_cols:
        raise ValueError(f"news file missing columns: {missing_news_cols}")

    tsm_symbol = find_tsm_symbol(prices)
    tsm_prices = prices.loc[
        prices['symbol'].astype(str) == tsm_symbol, ['date', price_col]
    ].dropna().sort_values('date')

    if tsm_prices.empty:
        raise ValueError(f"no rows for symbol = {tsm_symbol}")

    end_date = tsm_prices['date'].max()
    start_date = end_date - pd.DateOffset(years = LOOKBACK_YEARS)
    tsm_window = tsm_prices[
        (tsm_prices['date'] >= start_date) & (tsm_prices['date'] <= end_date)
    ].copy()

    daily_news = aggregate_daily_news(news)
    daily_window = daily_news[
        (daily_news['date'] >= start_date) & (daily_news['date'] <= end_date)
    ].copy()

    min_articles = int(os.getenv('MIN_ARTICLES', str(DEFAULT_MIN_ARTICLES)))
    top_n = int(os.getenv('TOP_HAWKISH_DAYS', str(DEFAULT_TOP_HAWKISH_DAYS)))
    hawkish_days = select_hawkish_days(daily_window, min_articles, top_n)

    plt.figure(figsize = (14, 6))
    plt.plot(tsm_window['date'], tsm_window[price_col], linewidth = 1.5)
    for hawkish_date in hawkish_days:
        plt.axvline(hawkish_date, linestyle = '--', color = 'red', linewidth = 1, alpha = 0.35)
    plt.title(f'{tsm_symbol} price (5Y) with top hawkish dates (dashed red)')
    plt.xlabel('Date')
    plt.ylabel(price_col)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi = PLOT_DPI)
    plt.close()

    print(f"saved {OUT_PATH.name}")
    print(f"symbol used: {tsm_symbol}, hawkish days plotted: {len(hawkish_days)}")


if __name__ == '__main__':
    main()
