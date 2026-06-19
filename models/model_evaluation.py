"""Run validation diagnostics: ROC, PR, calibration, top-k, walk-forward, etc."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'merged_features.parquet'
LABELS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.parquet'
VIS_DIR = PROJECT_ROOT / 'data' / 'visualisations'

PREFERRED_LABEL_COLUMNS = ['target_label', 'label', 'y', 'signal', 'class']
LABEL_TO_BINARY = {
    'SELL': 0, 'BUY': 1,
    'LOW_VOL': 0, 'HIGH_VOL': 1,
}

TRAIN_FRACTION = 0.7
CALIBRATION_FRACTION = 0.85
RANDOM_STATE = 42
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3
N_CALIBRATION_BINS = 10
N_WALK_FORWARD_SPLITS = 5
TOP_K_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)
PLA_EVENT_DATES = ['2024-05-23', '2024-10-14', '2025-04-01', '2025-12-29']
EVENT_VOL_WINDOW = 5


def load_and_label():
    """Load merged features and labels, merge them, and drop missing labels."""
    df_features = pd.read_parquet(FEATURES_PATH).assign(date = lambda frame: pd.to_datetime(frame['date']))
    df_labels = pd.read_parquet(LABELS_PATH).assign(date = lambda frame: pd.to_datetime(frame['date']))

    label_col = next((column for column in PREFERRED_LABEL_COLUMNS if column in df_labels.columns), None)
    if label_col is None:
        raise ValueError(
            f"no label column found in {LABELS_PATH.name}, "
            f"expected one of {PREFERRED_LABEL_COLUMNS}"
        )

    df = pd.merge(
        df_features,
        df_labels[['date', 'symbol', label_col]],
        on = ['date', 'symbol'], how = 'inner',
    ).sort_values('date').reset_index(drop = True)

    df = df.dropna(subset = [label_col]).copy()
    print(f"label column: {label_col}, rows: {len(df)}")

    label_strings = df[label_col].astype(str).str.upper().str.strip()
    y_float = label_strings.map(LABEL_TO_BINARY).values
    if np.isnan(y_float).any():
        unmapped = sorted(set(label_strings.unique()) - set(LABEL_TO_BINARY.keys()))
        raise ValueError(f"unmapped label values: {unmapped}")
    return df, label_col, y_float.astype(int)


def split_data(df, y, label_col):
    """Split chronologically into train / calibration / test."""
    feature_cols = [
        column for column in df.columns
        if column not in ('date', 'symbol', label_col) and pd.api.types.is_numeric_dtype(df[column])
    ]
    feature_matrix = df[feature_cols].fillna(0.0)

    row_count = len(df)
    train_end = int(TRAIN_FRACTION * row_count)
    cal_end = int(CALIBRATION_FRACTION * row_count)

    return (
        feature_matrix.iloc[:train_end], y[:train_end],
        feature_matrix.iloc[train_end:cal_end], y[train_end:cal_end],
        feature_matrix.iloc[cal_end:], y[cal_end:],
        feature_matrix,
    )


def save_simple_plot(filename, plot_fn):
    """Helper to save a single matplotlib plot and close the figure."""
    plt.figure()
    plot_fn()
    plt.tight_layout()
    plt.savefig(VIS_DIR / filename)
    plt.close()


def main():
    """Run the full evaluation pipeline and write all diagnostic plots."""
    VIS_DIR.mkdir(parents = True, exist_ok = True)

    df, label_col, y = load_and_label()
    x_train, y_train, x_cal, y_cal, x_test, y_test, x_full = split_data(df, y, label_col)

    rf = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        class_weight = 'balanced_subsample',
        random_state = RANDOM_STATE,
        n_jobs = -1,
    )
    rf.fit(x_train, y_train)

    calibrated = CalibratedClassifierCV(FrozenEstimator(rf), method = 'isotonic')
    calibrated.fit(x_cal, y_cal)
    p_test = calibrated.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, p_test)
    roc_auc = auc(fpr, tpr)
    save_simple_plot('roc_curve.png', lambda: (
        plt.plot(fpr, tpr),
        plt.plot([0, 1], [0, 1], linestyle = '--'),
        plt.title(f'ROC (AUC = {roc_auc:.3f})'),
        plt.xlabel('FPR'), plt.ylabel('TPR'),
    ))

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    avg_precision = average_precision_score(y_test, p_test)
    save_simple_plot('pr_curve.png', lambda: (
        plt.plot(rec, prec),
        plt.title(f'PR Curve (AP = {avg_precision:.3f})'),
        plt.xlabel('Recall'), plt.ylabel('Precision'),
    ))

    prob_true, prob_pred = calibration_curve(
        y_test, p_test, n_bins = N_CALIBRATION_BINS, strategy = 'quantile'
    )
    save_simple_plot('calibration_curve.png', lambda: (
        plt.plot(prob_pred, prob_true, marker = 'o'),
        plt.plot([0, 1], [0, 1], linestyle = '--'),
        plt.title('Calibration Curve'),
        plt.xlabel('Predicted Probability'), plt.ylabel('Observed Frequency'),
    ))

    sorted_y = y_test[np.argsort(-p_test)]
    base_rate = y_test.mean()
    top_k_rows = []
    for fraction in TOP_K_FRACTIONS:
        top_k = max(1, int(len(y_test) * fraction))
        top_k_rows.append((fraction, top_k, sorted_y[:top_k].mean()))
    df_topk = pd.DataFrame(top_k_rows, columns = ['fraction', 'k', 'precision'])
    df_topk.to_parquet(VIS_DIR / 'precision_at_topk.parquet', index = False)
    save_simple_plot('precision_at_topk.png', lambda: (
        plt.plot(df_topk['fraction'], df_topk['precision'], marker = 'o'),
        plt.axhline(base_rate, linestyle = '--'),
        plt.title('Precision @ Top-K'),
        plt.xlabel('Top fraction'), plt.ylabel('Precision'),
    ))

    ev_rows = []
    for thresh in THRESHOLD_GRID:
        take = p_test >= thresh
        ev_rows.append((thresh, np.nan if take.sum() == 0 else y_test[take].mean()))
    df_ev = pd.DataFrame(ev_rows, columns = ['threshold', 'mean_label'])
    df_ev.to_parquet(VIS_DIR / 'expected_value_vs_threshold.parquet', index = False)
    save_simple_plot('expected_value_vs_threshold.png', lambda: (
        plt.plot(df_ev['threshold'], df_ev['mean_label'], marker = 'o'),
        plt.title('EV proxy vs threshold'),
        plt.xlabel('Probability threshold'), plt.ylabel('Mean outcome'),
    ))

    cv_splitter = TimeSeriesSplit(n_splits = N_WALK_FORWARD_SPLITS)
    wf_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(x_full), start = 1):
        rf.fit(x_full.iloc[train_idx], y[train_idx])
        fold_p = rf.predict_proba(x_full.iloc[test_idx])[:, 1]
        wf_rows.append((fold, fold_p.mean()))
    df_wf = pd.DataFrame(wf_rows, columns = ['fold', 'mean_probability'])
    df_wf.to_parquet(VIS_DIR / 'walk_forward_metrics.parquet', index = False)
    save_simple_plot('walk_forward_metrics.png', lambda: (
        plt.plot(df_wf['fold'], df_wf['mean_probability'], marker = 'o'),
        plt.title('Walk-forward validation'),
        plt.xlabel('Fold'), plt.ylabel('Mean predicted probability'),
    ))

    tree_probs = np.stack(
        [tree.predict_proba(x_test)[:, 1] for tree in rf.estimators_], axis = 0
    )
    p_std = tree_probs.std(axis = 0)

    plt.figure(figsize = (12, 5))
    plt.plot(p_test, label = 'Mean Probability')
    plt.fill_between(
        range(len(p_test)),
        np.clip(p_test - p_std, 0, 1),
        np.clip(p_test + p_std, 0, 1),
        alpha = 0.3,
    )
    plt.title('Prediction Uncertainty Band')
    plt.xlabel('Test Observation')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'uncertainty_band_example.png')
    plt.close()

    try:
        run_event_analysis(df)
    except Exception as exc:
        print(f"event analysis skipped: {exc}")

    print(f"all plots written to {VIS_DIR.name}")


def run_event_analysis(df):
    """Compute pre/post realised volatility around PLA event dates for one symbol."""
    event_dates = pd.to_datetime(PLA_EVENT_DATES)
    upper_symbols = df['symbol'].astype(str).str.upper()
    sym = 'TSM' if 'TSM' in set(upper_symbols) else str(df['symbol'].iloc[0])

    symbol_group = df[upper_symbols == sym.upper()].copy().sort_values('date').reset_index(drop = True)
    if 'adj_close' not in symbol_group.columns or len(symbol_group) <= 30:
        return

    symbol_group['ret_1d'] = symbol_group['adj_close'].pct_change(fill_method = None)
    symbol_group['vol_pre_5d'] = symbol_group['ret_1d'].rolling(EVENT_VOL_WINDOW, min_periods = EVENT_VOL_WINDOW).std().shift(1)

    forward_base = symbol_group['ret_1d'].shift(-1)
    symbol_group['vol_post_5d'] = forward_base.rolling(
        EVENT_VOL_WINDOW, min_periods = EVENT_VOL_WINDOW
    ).std().shift(-(EVENT_VOL_WINDOW - 1))

    rows = []
    for date in event_dates:
        normalised = date.normalize()
        hit = symbol_group[symbol_group['date'] == normalised]
        if hit.empty:
            continue
        record = hit.iloc[0]
        if pd.isna(record['vol_pre_5d']) or pd.isna(record['vol_post_5d']):
            continue
        rows.append({
            'date': normalised.date(),
            'symbol': sym,
            'vol_pre_5d': float(record['vol_pre_5d']),
            'vol_post_5d': float(record['vol_post_5d']),
            'delta': float(record['vol_post_5d'] - record['vol_pre_5d']),
        })

    if not rows:
        return

    df_events = pd.DataFrame(rows)
    df_events.to_parquet(VIS_DIR / 'pla_event_vol_pre_post.parquet', index = False)

    x_positions = np.arange(len(df_events))
    width = 0.35
    plt.figure(figsize = (12, 5))
    plt.bar(x_positions - width/2, df_events['vol_pre_5d'], width, label = 'Pre (prev 5d)')
    plt.bar(x_positions + width/2, df_events['vol_post_5d'], width, label = 'Post (next 5d)')
    plt.xticks(x_positions, [str(date_value) for date_value in df_events['date']], rotation = 45, ha = 'right')
    plt.title(f'PLA exercise dates: realised vol pre vs post ({sym})')
    plt.xlabel('PLA exercise start date')
    plt.ylabel('Realised volatility (std of daily returns)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'pla_event_vol_pre_post.png')
    plt.close()


if __name__ == '__main__':
    main()
