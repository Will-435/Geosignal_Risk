"""Run every pipeline stage in order, skipping any missing or failing scripts."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    'pipelines/fetch_news.py',
    'pipelines/generate_stock_prices.py',
    'pipelines/process_rhetoric.py',
    'pipelines/merge_market_data.py',
    'utils/finance_metrics.py',
    'models/train_models.py',
    'pipelines/generate_signals.py',
    'pipelines/backtesting.py',
    'pipelines/PLA_effect.py',
    'models/risk_surface.py',
]


def main():
    """Iterate through SCRIPTS, running each in turn from the project root."""
    for relative in SCRIPTS:
        path = PROJECT_ROOT / relative
        if not path.exists():
            print(f"missing: {relative}")
            continue
        print(f"running: {relative}")
        subprocess.run([sys.executable, str(path)], check = False, cwd = PROJECT_ROOT)


if __name__ == '__main__':
    main()
