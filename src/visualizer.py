"""
visualizer.py
-------------
All matplotlib / seaborn visualisation logic lives here.

EDA plots
  - plot_closing_prices()
  - plot_volumes()
  - plot_moving_averages()
  - plot_daily_returns()
  - plot_correlation_heatmaps()
  - plot_pairplot()

LSTM result plots
  - plot_close_history()
  - plot_predictions()
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Global style (matches original notebook)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# ─────────────────────────────────────────────────────────────────────────────
# EDA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_closing_prices(company_list: list[pd.DataFrame]) -> None:
    """4-panel grid of closing price series for each ticker."""
    fig, _ = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, (company, ticker) in enumerate(zip(company_list, config.TECH_LIST), 1):
        ax = fig.add_subplot(2, 2, i)
        company["Close"].plot(ax=ax)
        ax.set_ylabel("Close")
        ax.set_xlabel(None)
        ax.set_title(f"Closing Price of {ticker}")

    plt.tight_layout()
    plt.show()


def plot_volumes(company_list: list[pd.DataFrame]) -> None:
    """4-panel grid of trading volume for each ticker."""
    fig, _ = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, (company, ticker) in enumerate(zip(company_list, config.TECH_LIST), 1):
        ax = fig.add_subplot(2, 2, i)
        company["Volume"].plot(ax=ax)
        ax.set_ylabel("Volume")
        ax.set_xlabel(None)
        ax.set_title(f"Sales Volume for {ticker}")

    plt.tight_layout()
    plt.show()


def plot_moving_averages(company_list: list[pd.DataFrame]) -> None:
    """4-panel grid showing Close + all configured moving averages."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    ma_cols = ["Close"] + [f"MA for {w} days" for w in config.MOVING_AVG_WINDOWS]
    titles = [config.COMPANY_NAMES[i] for i in range(len(config.TECH_LIST))]

    for ax, company, title in zip(axes.flatten(), company_list, titles):
        company[ma_cols].plot(ax=ax)
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def plot_daily_returns(company_list: list[pd.DataFrame]) -> None:
    """4-panel dot-and-line chart of each ticker's daily percentage change."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for ax, company, title in zip(
        axes.flatten(), company_list, config.COMPANY_NAMES
    ):
        company["Daily Return"].plot(
            ax=ax, legend=True, linestyle="--", marker="o"
        )
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def plot_pairplot(tech_rets: pd.DataFrame) -> None:
    """Seaborn pairplot of daily return correlations."""
    sns.pairplot(tech_rets, kind="reg")
    plt.show()


def plot_correlation_heatmaps(
    tech_rets: pd.DataFrame,
    closing_df: pd.DataFrame,
) -> None:
    """Side-by-side heatmaps: return correlation & closing-price correlation."""
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap="summer")
    plt.title("Correlation of stock return")

    plt.subplot(2, 2, 2)
    sns.heatmap(closing_df.corr(), annot=True, cmap="summer")
    plt.title("Correlation of stock closing price")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM result plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_close_history(df: pd.DataFrame) -> None:
    """Simple line chart of the primary ticker's full Close price history."""
    plt.figure(figsize=(16, 6))
    plt.title(f"{config.PRIMARY_TICKER_NAME} Close Price History")
    plt.plot(df["Close"])
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.show()


def plot_predictions(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    future_df: pd.DataFrame | None = None,
) -> None:
    """
    Overlay train, validation actuals, LSTM predictions, RF predictions
    (if present), and the optional future forecast on a single chart.

    Expected columns in `valid`
    ---------------------------
    'Close'            : actual closing prices  (required)
    'Predictions'      : LSTM predictions       (required)
    'RF Predictions'   : RF predictions         (optional — plotted when present)

    Parameters
    ----------
    train      : training slice of the original DataFrame
    valid      : validation slice with prediction columns already attached
    future_df  : optional DataFrame with a 'Future Predictions' column
    """
    plt.figure(figsize=(16, 8))

    title = f"{config.PRIMARY_TICKER_NAME} Stock Price Prediction — LSTM vs Random Forest"
    if future_df is not None:
        title += f" (with {config.NUM_FUTURE_DAYS}-Day Forecast)"
    plt.title(title)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)

    # ── Actuals ───────────────────────────────────────────────────────────────
    plt.plot(train["Close"],  color="blue",  label="Training Data (Actual)")
    plt.plot(valid["Close"],  color="green", label="Validation Data (Actual)")

    # ── LSTM predictions ──────────────────────────────────────────────────────
    if "Predictions" in valid.columns:
        plt.plot(
            valid["Predictions"],
            color="red",
            linewidth=1.5,
            label="LSTM Predictions",
        )

    # ── RF predictions (optional) ─────────────────────────────────────────────
    if "RF Predictions" in valid.columns:
        plt.plot(
            valid["RF Predictions"],
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label="Random Forest Predictions",
        )

    # ── Future forecast (optional) ────────────────────────────────────────────
    if future_df is not None:
        plt.plot(
            future_df["Future Predictions"],
            color="purple",
            linestyle="--",
            label=f"Future {config.NUM_FUTURE_DAYS} Days (LSTM Forecast)",
        )

    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
