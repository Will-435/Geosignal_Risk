import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.nlp_helpers import apply_vader, apply_finbert


INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'multisource_headlines.csv'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'headlines_with_sentiment.csv'


def main():
    """Score every headline with VADER and FinBERT, then save."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"input file not found at {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df["title"] = df["title"].fillna("")

    df = apply_vader(df)
    df = apply_finbert(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"sentiment scored: {len(df)} rows")


if __name__ == "__main__":
    main()
