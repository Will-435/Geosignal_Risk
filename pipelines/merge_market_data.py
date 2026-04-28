from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEADLINES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'headlines_with_sentiment.csv'
PRICES_PATH = PROJECT_ROOT / 'data' / 'raw' / 'stock_prices.csv'
TENSION_PATH = PROJECT_ROOT / 'data' / 'processed' / 'satellite_tension_scores.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'merged_features.csv'


def main():
    """Aggregate daily sentiment and merge with prices and (optional) tension scores."""
    df_news = pd.read_csv(HEADLINES_PATH)
    df_prices = pd.read_csv(PRICES_PATH)
    df_tension = pd.read_csv(TENSION_PATH) if TENSION_PATH.exists() else pd.DataFrame()

    daily_sentiment = df_news.groupby("date").agg({
        "vader_compound": "mean",
        "finbert_sentiment": (
            lambda values: values.mode().iloc[0] if not values.mode().empty else "neutral"
        )
    }).reset_index()

    merged = pd.merge(df_prices, daily_sentiment, on="date", how="left")
    if not df_tension.empty:
        merged = pd.merge(merged, df_tension, on="date", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"merged features saved: {len(merged)} rows")


if __name__ == "__main__":
    main()
