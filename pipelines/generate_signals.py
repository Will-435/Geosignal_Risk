"""Score the latest available date and emit DE_RISK / RISK_ON / HOLD per ticker."""

import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / 'models' / 'signal_generator' / 'rf_signal_model.pkl'
FEATURES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'merged_features.parquet'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'signals_latest.parquet'

DEFAULT_RISK_OFF_THRESH = 0.60
DEFAULT_RISK_ON_THRESH = 0.40
DEFAULT_LOOKBACK_DAYS = 14
PROBABILITY_MIDPOINT = 0.5

RISK_OFF_THRESH = float(os.environ.get('RISK_OFF_THRESH', DEFAULT_RISK_OFF_THRESH))
RISK_ON_THRESH = float(os.environ.get('RISK_ON_THRESH', DEFAULT_RISK_ON_THRESH))


def load_inputs():
    """Load the trained model and the merged feature set, normalising key columns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model file not found: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"merged feature file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("expected 'date' and 'symbol' in merged_features.parquet")
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return model, df


def feature_columns(df):
    """Return the union of text and market feature columns present in df."""
    text_features = ["vader_compound"] + [column for column in df.columns if column.endswith("_tension")]
    market_features = [column for column in df.columns if column.startswith(("ret_", "vol_", "ma_"))]
    return [column for column in (text_features + market_features) if column in df.columns]


def pick_latest_with_rows(df, lookback_days = DEFAULT_LOOKBACK_DAYS):
    """Return the most recent date in df that actually has rows (handles weekends)."""
    max_date = df["date"].max()
    if pd.isna(max_date):
        return None
    for offset in range(0, lookback_days + 1):
        candidate = (max_date - timedelta(days = offset)).date()
        if (df["date"].dt.date == candidate).any():
            return pd.Timestamp(candidate)
    return None


def prob_to_signal(probability):
    """Map a HIGH_VOL probability to DE_RISK / RISK_ON / HOLD."""
    if probability >= RISK_OFF_THRESH:
        return 'DE_RISK'
    if probability <= RISK_ON_THRESH:
        return 'RISK_ON'
    return 'HOLD'


def main():
    """Load model + features, score the latest available date, save and print signals."""
    model, df = load_inputs()
    base_features = feature_columns(df)

    scoring_date = pick_latest_with_rows(df)
    if scoring_date is None:
        print("no rows found in last 14 days")
        raise SystemExit(0)

    df_day = df[df["date"].dt.date == scoring_date.date()].copy()
    if df_day.empty:
        print("scoring date has no rows after date-only match")
        raise SystemExit(0)

    df_day[base_features] = df_day[base_features].fillna(0.0)
    scorable = df_day[["symbol"] + base_features].groupby("symbol", as_index = False).mean()

    symbol_dummies = pd.get_dummies(scorable["symbol"], prefix = "sym")
    scorable = pd.concat([scorable, symbol_dummies], axis = 1)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = base_features + [column for column in scorable.columns if column.startswith("sym_")]

    feature_matrix = scorable.reindex(columns = expected, fill_value = 0.0)

    try:
        proba = model.predict_proba(feature_matrix)
        p_high_vol = proba[:, 1]
    except Exception:
        predictions = model.predict(feature_matrix)
        p_high_vol = (predictions == 1).astype(float)

    signals = pd.DataFrame({
        "date": scoring_date.date().isoformat(),
        "symbol": scorable["symbol"],
        "p_high_vol": p_high_vol,
    })
    signals["signal"] = signals["p_high_vol"].apply(prob_to_signal)
    signals["confidence"] = (signals["p_high_vol"] - PROBABILITY_MIDPOINT).abs() * 2

    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    signals[["date", "symbol", "signal", "p_high_vol", "confidence"]].to_parquet(
        OUTPUT_PATH, index = False
    )

    print(f"date scored: {scoring_date.date()}")
    for _, row in signals.iterrows():
        print(
            f"  {row['symbol']}: {row['signal']} "
            f"(p={row['p_high_vol']:.2f}, conf={row['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
