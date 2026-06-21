import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from utils.config import NEWSAPI_KEY, NEWSAPI_AI_KEY, MEDIASTACK_KEY, GNEWS_KEY
except ImportError:
    NEWSAPI_KEY = ""
    NEWSAPI_AI_KEY = ""
    MEDIASTACK_KEY = ""
    GNEWS_KEY = ""
# Paste your keys here using the links in README.md

NEWSAPI_PAGE_SIZE = 20
NEWSAPI_AI_COUNT = 20
MEDIASTACK_LIMIT = 20
GNEWS_MAX = 20

OUTPUT_REL_PATH = Path('data') / 'raw' / 'multisource_headlines.parquet'


def _today():
    return datetime.today().strftime('%Y-%m-%d')


def fetch_newsapi():
    """Fetch defence/Taiwan/military headlines from NewsAPI."""
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    url = (
        f"https://newsapi.org/v2/everything?q=defense+OR+military+OR+Taiwan"
        f"&language=en&sortBy=publishedAt&pageSize={NEWSAPI_PAGE_SIZE}"
        f"&apiKey={NEWSAPI_KEY}"
    )
    response = requests.get(url)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    articles = response.json().get("articles", [])
    today = _today()
    rows = [
        {"source": "NewsAPI", "title": article["title"], "url": article["url"], "date": today}
        for article in articles if article.get("title")
    ]
    return pd.DataFrame(rows)


def fetch_newsapi_ai():
    """Fetch defence/Taiwan/military headlines from NewsAPI.ai (Event Registry)."""
    if not NEWSAPI_AI_KEY:
        return pd.DataFrame()
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": ["military", "defense", "Taiwan", "China"],
        "keywordOper": "or",
        "lang": "eng",
        "articlesSortBy": "date",
        "articlesCount": NEWSAPI_AI_COUNT,
        "resultType": "articles",
        "apiKey": NEWSAPI_AI_KEY,
    }
    response = requests.post(url, json=payload)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    articles = response.json().get("articles", {}).get("results", [])
    today = _today()
    rows = [
        {"source": "NewsAPI.ai", "title": article["title"], "url": article["url"], "date": today}
        for article in articles if article.get("title")
    ]
    return pd.DataFrame(rows)


def fetch_mediastack():
    """Fetch defence/Taiwan/military headlines from Mediastack."""
    if not MEDIASTACK_KEY:
        return pd.DataFrame()
    url = (
        f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_KEY}"
        f"&keywords=defense,military,Taiwan,China&languages=en&limit={MEDIASTACK_LIMIT}"
    )
    response = requests.get(url)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    articles = response.json().get("data", [])
    today = _today()
    rows = [
        {"source": "Mediastack", "title": article["title"], "url": article["url"], "date": today}
        for article in articles if article.get("title")
    ]
    return pd.DataFrame(rows)


def fetch_gnews():
    """Fetch defence/Taiwan/military headlines from GNews."""
    if not GNEWS_KEY:
        return pd.DataFrame()
    url = (
        f"https://gnews.io/api/v4/search?q=military+OR+defense+OR+China+Taiwan"
        f"&lang=en&country=us&max={GNEWS_MAX}&token={GNEWS_KEY}"
    )
    response = requests.get(url)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    articles = response.json().get("articles", [])
    today = _today()
    rows = [
        {"source": "GNews", "title": article["title"], "url": article["url"], "date": today}
        for article in articles if article.get("title")
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / OUTPUT_REL_PATH

    frames = []
    for label, fetcher in [
        ("NewsAPI", fetch_newsapi),
        ("NewsAPI.ai", fetch_newsapi_ai),
        ("Mediastack", fetch_mediastack),
        ("GNews", fetch_gnews),
        ]:
        try:
            frame = fetcher()
            if not frame.empty:
                frames.append(frame)
                print(f"{label}: {len(frame)} articles")
            else:
                print(f"{label}: skipped (no key or no results)")
        except Exception as exc:
            print(f"{label} failed: {exc}")

    if frames:
        merged = pd.concat(frames).drop_duplicates(subset = ["title", "url"])
        output_path.parent.mkdir(parents = True, exist_ok = True)
        merged.to_parquet(output_path, index = False)
        print(f"saved {len(merged)} headlines")
    else:
        print("no headlines collected")
