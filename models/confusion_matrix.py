"""Compare model risk-off decisions against the true volatility regime."""

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTUAL_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.parquet'
PREDICTED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'backtest_vol_strategy.parquet'
OUT_PATH = PROJECT_ROOT / 'data' / 'visualisations' / 'risk_confusion_matrix.png'

LABEL_TO_BINARY = {'HIGH_VOL': 1, 'LOW_VOL': 0}
PLOT_DPI = 300


def main():
    """Build and save the confusion matrix plot, print classification report."""
    OUT_PATH.parent.mkdir(parents = True, exist_ok = True)

    df_actual = pd.read_parquet(ACTUAL_PATH)
    df_preds = pd.read_parquet(PREDICTED_PATH)

    df_actual['date'] = pd.to_datetime(df_actual['date'])
    df_preds['date'] = pd.to_datetime(df_preds['date'])

    df_actual['actual_risk'] = df_actual['target_label'].map(LABEL_TO_BINARY)

    comparison = pd.merge(
        df_actual[['date', 'actual_risk']],
        df_preds[['date', 'risk_off']],
        on = 'date',
    ).dropna()

    cm = confusion_matrix(comparison['actual_risk'], comparison['risk_off'])

    plt.figure(figsize = (8, 6))
    sns.heatmap(
        cm, annot = True, fmt = 'd', cmap = 'Blues',
        xticklabels = ['Predicted: LOW', 'Predicted: HIGH'],
        yticklabels = ['Actual: LOW', 'Actual: HIGH'],
    )
    plt.title('Volatility Pipeline: Confusion Matrix', fontsize = 14)
    plt.ylabel('Ground Truth (Actual Regime)')
    plt.xlabel('Model Decision (Risk-Off Trigger)')
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi = PLOT_DPI)
    plt.close()

    print(f"saved confusion matrix to {OUT_PATH.name}")
    print(classification_report(comparison['actual_risk'], comparison['risk_off']))


if __name__ == '__main__':
    main()
