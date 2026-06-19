"""Train a ticker-aware Random Forest classifier on text + market features."""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MERGED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'merged_features.parquet'
LABELED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.parquet'
MODEL_OUT = PROJECT_ROOT / 'models' / 'signal_generator' / 'rf_signal_model.pkl'
METRICS_OUT = PROJECT_ROOT / 'models' / 'signal_generator' / 'metrics_per_symbol.parquet'

USE_THRESHOLD_LABELS = os.environ.get("USE_THRESHOLD_LABELS", "0") == "1"
FORWARD_HORIZON_DAYS = int(os.environ.get("FUT_H", "1"))
FORWARD_RETURN_THRESHOLD = float(os.environ.get("FUT_RET_THRESH", "0.003"))
TUNE_RF = os.environ.get("TUNE_RF", "0") == "1"

LABEL_CANDIDATE_COLUMNS = ["target_label", "label", "target", "signal", "class", "direction", "y"]
TRAIN_FRACTION = 0.8
MIN_SAMPLES_PER_SYMBOL = 5
TUNING_ITERATIONS = 10
RANDOM_STATE = 42

BASE_RF_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",
    "min_samples_leaf": 3,
}

TUNING_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [2, 4, 6],
    "class_weight": ["balanced", "balanced_subsample"],
}

POSITIVE_LABELS = {
    'buy', 'long', 'up', 'positive', 'bull', 'increase', '1', 'true',
    'high_vol', 'highvol', 'high-vol', 'high volatility', 'high_volatility',
}
NEGATIVE_LABELS = {
    'sell', 'short', 'down', 'negative', 'bear', 'decrease', 'false', '0',
    'low_vol', 'lowvol', 'low-vol', 'low volatility', 'low_volatility',
}


def normalise_symbol(series):
    """Trim and uppercase a symbol column."""
    return series.astype(str).str.strip().str.upper()


def compute_forward_labels(frame, horizon, threshold):
    """Label rows BUY / SELL based on the next-period adj_close return."""
    if "adj_close" not in frame.columns:
        raise ValueError("adj_close required to compute threshold labels")
    tmp = frame.sort_values(["symbol", "date"], kind = "mergesort").copy()
    forward_price = tmp.groupby("symbol")["adj_close"].shift(-horizon)
    forward_return = (forward_price / tmp["adj_close"]) - 1.0
    out = pd.Series(index = tmp.index, dtype = "object")
    out[forward_return > threshold] = "BUY"
    out[forward_return < -threshold] = "SELL"
    return out.reindex(frame.index)


def map_existing_labels_to_binary(series):
    """Map a string label column onto {0.0, 1.0} using the positive/negative sets."""
    cleaned = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index = cleaned.index, dtype = 'float')
    out.loc[cleaned.isin(POSITIVE_LABELS)] = 1.0
    out.loc[cleaned.isin(NEGATIVE_LABELS)] = 0.0
    return out


def load_and_merge():
    """Load both feature files and merge on date+symbol, keeping required columns."""
    df_features = pd.read_parquet(MERGED_PATH)
    df_labels = pd.read_parquet(LABELED_PATH)

    for frame in (df_features, df_labels):
        if 'date' in frame.columns:
            frame['date'] = pd.to_datetime(frame['date'], errors = "coerce").dt.date

    df_features["symbol"] = normalise_symbol(df_features["symbol"])
    df_labels["symbol"] = normalise_symbol(df_labels["symbol"])

    text_features = ["vader_compound"] + [column for column in df_features.columns if column.endswith("_tension")]
    market_features = [column for column in df_labels.columns if column.startswith(("ret_", "vol_", "ma_"))]

    missing_text = [column for column in text_features if column not in df_features.columns]
    if missing_text:
        raise ValueError(f"merged_features.parquet missing NLP columns: {missing_text}")

    label_keep = ["date", "symbol"]
    if "adj_close" in df_labels.columns:
        label_keep.append("adj_close")
    have_labels = [column for column in LABEL_CANDIDATE_COLUMNS if column in df_labels.columns]
    label_keep += have_labels + market_features

    feature_keep = ["date", "symbol"] + text_features
    if "adj_close" in df_features.columns:
        feature_keep.append("adj_close")

    merged = pd.merge(
        df_features[feature_keep],
        df_labels[label_keep].drop_duplicates(["date", "symbol"]),
        on = ["date", "symbol"],
        how = "inner",
    )
    return merged, text_features, market_features


def select_label_series(df):
    """Return a string label series, either from threshold rule or an existing column."""
    if USE_THRESHOLD_LABELS:
        print(f"using threshold labels (h={FORWARD_HORIZON_DAYS}, "
              f"thresh={FORWARD_RETURN_THRESHOLD:.4f})")
        return compute_forward_labels(df, FORWARD_HORIZON_DAYS, FORWARD_RETURN_THRESHOLD)

    label_col = next((column for column in LABEL_CANDIDATE_COLUMNS if column in df.columns), None)
    if label_col is None:
        print("no explicit label column, falling back to threshold labels")
        return compute_forward_labels(df, FORWARD_HORIZON_DAYS, FORWARD_RETURN_THRESHOLD)

    print(f"using label column: {label_col}")
    return df[label_col]


def fit_classifier(train_x, train_y):
    """Train base RF or run RandomizedSearchCV with TimeSeriesSplit if TUNE_RF=1."""
    base_clf = RandomForestClassifier(**BASE_RF_PARAMS)

    if not TUNE_RF:
        base_clf.fit(train_x, train_y)
        return base_clf

    print("running RandomizedSearchCV...")
    cv_splitter = TimeSeriesSplit(n_splits = 3)
    search = RandomizedSearchCV(
        estimator = base_clf,
        param_distributions = TUNING_GRID,
        n_iter = TUNING_ITERATIONS,
        cv = cv_splitter,
        random_state = RANDOM_STATE,
        n_jobs = -1,
        verbose = 1,
    )
    search.fit(train_x, train_y)
    print(f"best params: {search.best_params_}")
    return search.best_estimator_


def per_symbol_metrics(test_part, classifier, feature_cols):
    """Compute classification metrics broken down by symbol for the test set."""
    rows = []
    for symbol, group in test_part.groupby("symbol"):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            continue
        truth = group["label"].astype(int)
        prediction = classifier.predict(group[feature_cols])
        report = classification_report(
            truth, prediction, digits = 4, zero_division = 0, output_dict = True
        )
        rows.append({
            "symbol": symbol,
            "support_0": int(report.get("0", {}).get("support", 0)),
            "precision_0": report.get("0", {}).get("precision", np.nan),
            "recall_0": report.get("0", {}).get("recall", np.nan),
            "f1_0": report.get("0", {}).get("f1-score", np.nan),
            "support_1": int(report.get("1", {}).get("support", 0)),
            "precision_1": report.get("1", {}).get("precision", np.nan),
            "recall_1": report.get("1", {}).get("recall", np.nan),
            "f1_1": report.get("1", {}).get("f1-score", np.nan),
            "accuracy": report.get("accuracy", np.nan),
        })
    return rows


def main():
    """Train the model end-to-end and persist the classifier and per-symbol metrics."""
    MODEL_OUT.parent.mkdir(parents = True, exist_ok = True)

    df, text_features, market_features = load_and_merge()
    label_series = select_label_series(df)

    df[text_features] = df[text_features].fillna(0.0)
    if market_features:
        df[market_features] = df[market_features].fillna(0.0)

    symbol_dummies = pd.get_dummies(df["symbol"], prefix = "sym")
    df = pd.concat([df, symbol_dummies], axis = 1)

    feature_cols = text_features + market_features + list(symbol_dummies.columns)

    binary_labels = map_existing_labels_to_binary(label_series)
    train_df = pd.concat(
        [df[["date", "symbol"]], df[feature_cols], binary_labels.rename("label")],
        axis = 1,
    )

    before = len(train_df)
    train_df = train_df.dropna(subset = ["label"])
    after = len(train_df)
    print(f"kept {after} rows (dropped {before - after} unlabelled)")
    if after == 0:
        raise RuntimeError("no rows left after label filtering")

    train_df = train_df.sort_values("date", kind = "mergesort")
    split_index = int(len(train_df) * TRAIN_FRACTION)
    train_part = train_df.iloc[:split_index].copy()
    test_part = train_df.iloc[split_index:].copy()

    classifier = fit_classifier(train_part[feature_cols], train_part["label"].astype(int))

    print(classification_report(
        test_part["label"].astype(int),
        classifier.predict(test_part[feature_cols]),
        digits = 4, zero_division = 0,
    ))

    rows = per_symbol_metrics(test_part, classifier, feature_cols)
    if rows:
        pd.DataFrame(rows).to_parquet(METRICS_OUT, index = False)
    else:
        print("not enough samples per symbol to save metrics")

    joblib.dump(classifier, MODEL_OUT)
    print(f"model saved: {MODEL_OUT.name}")


if __name__ == '__main__':
    main()
