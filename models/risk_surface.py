"""Render an interactive 3D scatter of sentiment, volatility, and returns."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'data' / 'visualisations'
LABELED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'labeled_features.parquet'
BACKTEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'backtest_vol_strategy.parquet'

LABEL_TO_NUMERIC = {'LOW_VOL': 0, 'HIGH_VOL': 1, 'V_HIGH_VOL': 2, 'BUY': 0, 'SELL': 1}
PLOT_WIDTH = 1200
PLOT_HEIGHT = 800
MARKER_SIZE = 4


def build_figure(df, backtest):
    """Build the 3D figure combining a market scatter and a strategy mesh surface."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x = df['vader_compound'],
        y = df['vol_10d'],
        z = df['ret_1d'],
        mode = 'markers',
        marker = dict(
            size = MARKER_SIZE,
            color = df['target_label'].map(LABEL_TO_NUMERIC),
            colorscale = 'Viridis',
            opacity = 0.8,
            colorbar = dict(
                title = "Risk Regime",
                tickvals = [0, 1, 2],
                ticktext = ['Low', 'High', 'V. High'],
            ),
        ),
        text = df['date'].dt.strftime('%Y-%m-%d'),
        name = 'Market Data Points',
        hovertemplate = (
            "<b>Date:</b> %{text}<br>"
            "<b>Sentiment:</b> %{x}<br>"
            "<b>Vol:</b> %{y}<br>"
            "<b>Return:</b> %{z}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Mesh3d(
        x = backtest.index,
        y = backtest['p_high_vol'],
        z = backtest['strategy_equity'],
        opacity = 0.25,
        color = 'cyan',
        name = 'Strategy Equity Surface',
    ))

    fig.update_layout(
        updatemenus = [dict(
            type = "buttons", direction = "right", x = 0.7, y = 1.1, showactive = True,
            buttons = [
                dict(label = "Show All", method = "update",
                     args = [{"visible": [True, True]}]),
                dict(label = "Data Points Only", method = "update",
                     args = [{"visible": [True, False]}]),
                dict(label = "Surface Only", method = "update",
                     args = [{"visible": [False, True]}]),
            ],
        )],
        title = dict(
            text = "Sentiment, Volatility, and Returns",
            y = 0.9, x = 0.5, xanchor = 'center', yanchor = 'top',
        ),
        scene = dict(
            xaxis_title = 'Vader Sentiment Score',
            yaxis_title = '10D Realised Volatility',
            zaxis_title = '1D Log Returns',
            xaxis = dict(backgroundcolor = "rgb(20, 20, 20)", gridcolor = "gray", showbackground = True),
            yaxis = dict(backgroundcolor = "rgb(20, 20, 20)", gridcolor = "gray", showbackground = True),
            zaxis = dict(backgroundcolor = "rgb(20, 20, 20)", gridcolor = "gray", showbackground = True),
        ),
        template = 'plotly_dark',
        width = PLOT_WIDTH,
        height = PLOT_HEIGHT,
        margin = dict(l = 0, r = 0, b = 0, t = 50),
    )
    return fig


def main():
    """Build and save the risk-surface figure as PNG and HTML."""
    OUT_DIR.mkdir(parents = True, exist_ok = True)

    df = pd.read_parquet(LABELED_PATH)
    df['date'] = pd.to_datetime(df['date'])
    backtest = pd.read_parquet(BACKTEST_PATH)

    fig = build_figure(df, backtest)

    png_path = OUT_DIR / 'volatility_risk_surface.png'
    html_path = OUT_DIR / 'volatility_risk_surface.html'

    try:
        fig.write_image(str(png_path), scale = 2)
        print(f"saved {png_path.name}")
    except Exception as exc:
        print(f"png save failed (kaleido missing?): {exc}")

    fig.write_html(str(html_path))
    print(f"saved {html_path.name}")


if __name__ == '__main__':
    main()
