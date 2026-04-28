import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.news_scraper import fetch_newsapi, fetch_gnews, fetch_mediastack


OUTPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'multisource_headlines.csv'

FETCHERS = [
    ("NewsAPI", fetch_newsapi),
    ("GNews", fetch_gnews),
    ("Mediastack", fetch_mediastack),
]


def main():
    """Pull headlines from each configured news API and save the union."""
    frames = []
    for label, fetcher in FETCHERS:
        try:
            frame = fetcher()
            if not frame.empty:
                frames.append(frame)
                print(f"{label}: {len(frame)} articles")
            else:
                print(f"{label}: no data (key missing or empty result)")
        except Exception as exc:
            print(f"{label} failed: {exc}")

    if not frames:
        print("no headlines collected, skipping save")
        return

    merged = pd.concat(frames).drop_duplicates(subset=["title", "url"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"saved {len(merged)} headlines")


if __name__ == "__main__":
    main()
