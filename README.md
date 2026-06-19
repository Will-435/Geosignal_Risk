# GeoSignalRisk

A research pipeline that turns geopolitical news flow and PLA military activity into a forward-looking probability of a high-volatility regime in Taiwan-exposed equity markets, and uses that probability to dynamically scale exposure to TSMC.

## What this project predicts

GeoSignalRisk is built around a deliberate refusal to predict the direction of returns. Direction prediction in equities is an exceptionally noisy target: the signal-to-noise ratio is so low that even a strong model rarely produces a stable, economically meaningful edge once realistic transaction costs and execution are accounted for. The framing here is different. The pipeline asks the much more tractable question of whether the next five trading days are likely to be a stressed, high-volatility regime, given the prevailing news sentiment, geopolitical tension scores, and recent market state. That probability, rather than a directional forecast, is what drives the trading rule. When the probability of a high-volatility regime is high, exposure is reduced toward a configurable floor; when it is low, the strategy stays close to fully invested. The end result is a risk-aware overlay on top of a passive holding, designed to preserve the bulk of long-run returns while clipping the worst drawdowns associated with geopolitical shocks. The choice of Taiwan-exposed equities (primarily TSM, with INTC and Samsung as secondary references) reflects the underlying research interest in the PLA-Taiwan tension channel and its observable market signature.

## How the pipeline is structured

The pipeline is split into clearly demarcated stages, each driven by a single script. News headlines are pulled from three free-tier APIs (NewsAPI, GNews, Mediastack), market data from Yahoo Finance via yfinance, and sentiment is scored with both VADER (fast, lexicon-based) and FinBERT (transformer-based, finance-tuned). Daily sentiment is then aggregated, joined to price history, and converted into engineered features (lagged returns, rolling moving averages, rolling realised volatilities). The label, defined as "is forward 5-day realised volatility above its rolling 252-day 75th percentile?", is computed using past-only quantiles to avoid look-ahead leakage. A Random Forest classifier is trained on the joined dataset with TimeSeriesSplit cross-validation, and a separate evaluation script produces calibration curves, ROC, precision-at-top-k, walk-forward stability, and PLA event-window analyses.

## Repository layout

```
geosignal_risk/
+-- README.md
+-- requirements.txt
+-- run_all.py                 # orchestrates the full pipeline
+-- pipelines/                 # data acquisition, feature build, scoring, backtest
|   +-- fetch_news.py
|   +-- generate_stock_prices.py
|   +-- process_rhetoric.py
|   +-- merge_market_data.py
|   +-- generate_signals.py
|   +-- backtesting.py
|   +-- PLA_effect.py
+-- utils/                     # shared helpers
|   +-- news_scraper.py
|   +-- nlp_helpers.py
|   +-- finance_metrics.py
+-- models/                    # training and evaluation
|   +-- train_models.py
|   +-- model_evaluation.py
|   +-- confusion_matrix.py
|   +-- plot_tsmc_hawkish.py
|   +-- risk_surface.py
|   +-- signal_generator/
|       +-- rf_signal_model.pkl
+-- data/
    +-- raw/                   # cached headlines, prices
    +-- processed/             # sentiment, merged features, labels, backtest
    +-- visualisations/        # plots produced by the eval scripts
```

## A note on API keys

The three news API keys live in a single file, [utils/config.py](utils/config.py), as the global variables `NEWSAPI_KEY`, `GNEWS_KEY`, and `MEDIASTACK_KEY`. That file is gitignoredto avoid leakage of API keys. A committed template, [utils/config.example.py](utils/config.example.py), documents the expected variable names. Anyone wanting to reproduce the pipeline must source their own free-tier keys from NewsAPI, GNews, and Mediastack (all of which have a free tier sufficient for this project), copy `utils/config.example.py` to `utils/config.py`, and paste their keys in. With the keys absent or left as placeholders, `pipelines/fetch_news.py` is a graceful no-op: it logs that each fetcher is skipped, leaves the existing `data/raw/multisource_headlines.csv` cache intact, and the rest of the pipeline can still run end-to-end against that cached snapshot.

## Getting started

Install dependencies into a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then either run a single stage:

```bash
python pipelines/backtesting.py
```

or the whole thing in order:

```bash
python run_all.py
```

## What the pipeline produces

| Output | Meaning |
|--------|---------|
| `data/processed/headlines_with_sentiment.csv` | Each headline scored by VADER and FinBERT |
| `data/processed/merged_features.csv` | Daily prices joined to daily aggregated sentiment |
| `data/processed/labeled_features.csv` | The above with HIGH_VOL / LOW_VOL labels and market features |
| `data/processed/signals_latest.csv` | DE_RISK / RISK_ON / HOLD per ticker for the latest available date |
| `data/processed/backtest_vol_strategy.csv` | Backtest trace including p_high_vol, exposure, equity curves |
| `data/processed/pla_event_study_bp.csv` | TSMC's basis-point reaction around each major PLA exercise |
| `data/visualisations/*.png` | All diagnostic plots |

## Honest limitations

The training data is small and the headline corpus thin, so out-of-sample stability is modest and the backtest equity curve should be read with appropriate scepticism. The label definition (forward 5-day realised volatility above the rolling 75th percentile) is a sensible starting point but it is not the only defensible choice, and the threshold parameters interact with the data window in ways the current evaluation does not fully decompose. Sentiment scoring is also a known weak link: VADER is fast but generic, FinBERT is finance-aware but trained on quite different text, and neither captures the second-order signal of *who* is making a statement and to *whom*. Treat the output as a research artefact, not as production trading infrastructure.
