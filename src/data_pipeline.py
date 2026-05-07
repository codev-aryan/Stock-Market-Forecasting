"""
data_pipeline.py
----------------
Handles all data acquisition and preprocessing:
  - yfinance downloads
  - Moving averages & daily returns (EDA features)
  - MinMax scaling
  - 60-day lookback sequence creation
  - Train / test split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import config


# ─────────────────────────────────────────────────────────────────────────────
# EDA Data
# ─────────────────────────────────────────────────────────────────────────────

def download_eda_data() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Download 1-year price data for every ticker in config.TECH_LIST.

    Returns
    -------
    company_list : list of DataFrames (one per ticker, with MA & Daily Return cols)
    closing_df   : wide DataFrame of daily closing prices (for correlation plots)
    """
    company_list: list[pd.DataFrame] = []

    for ticker, name in zip(config.TECH_LIST, config.COMPANY_NAMES):
        df = yf.download(ticker, config.EDA_START_DATE, config.EDA_END_DATE)
        df["company_name"] = name

        # Moving averages
        for window in config.MOVING_AVG_WINDOWS:
            df[f"MA for {window} days"] = df["Close"].rolling(window).mean()

        # Daily return
        df["Daily Return"] = df["Close"].pct_change()

        company_list.append(df)

    # Wide closing-price DataFrame for correlation analysis
    raw = yf.download(
        config.TECH_LIST,
        start=config.EDA_START_DATE,
        end=config.EDA_END_DATE,
    )
    closing_df = raw["Close"]

    return company_list, closing_df


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Data
# ─────────────────────────────────────────────────────────────────────────────

def download_lstm_data() -> pd.DataFrame:
    """Download long-history closing prices for the primary ticker."""
    df = yf.download(
        config.PRIMARY_TICKER,
        start=config.LSTM_START_DATE,
        end=config.LSTM_END_DATE,
    )
    print(f"\n--- {config.PRIMARY_TICKER_NAME} Stock Data (First 5 Rows) ---")
    print(df.head())
    print(f"\n--- {config.PRIMARY_TICKER_NAME} Stock Data (Last 5 Rows) ---")
    print(df.tail())
    print(f"\nDataFrame shape: {df.shape}")
    return df


def build_lstm_datasets(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, int, np.ndarray]:
    """
    Scale closing prices and create supervised-learning sequences.

    Parameters
    ----------
    df : raw OHLCV DataFrame from yfinance

    Returns
    -------
    x_train, y_train  : training sequences and targets
    x_test,  y_test   : test sequences and targets
    scaler            : fitted MinMaxScaler (needed to invert predictions)
    training_data_len : index boundary between train and test sets
    scaled_data       : full scaled array (used for future forecasting)
    """
    data: pd.DataFrame = df[["Close"]]
    dataset: np.ndarray = data.values

    training_data_len: int = int(np.ceil(len(dataset) * config.TRAIN_SPLIT_RATIO))

    # Scale using ONLY training data (prevents data leakage)
    train_data_raw = dataset[:training_data_len]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data_raw)

    scaled_data: np.ndarray = scaler.transform(dataset)

    # ── Training sequences ────────────────────────────────────────────────────
    train_data = scaled_data[:training_data_len, :]
    x_train: list = []
    y_train: list = []

    window = config.LOOKBACK_WINDOW
    for i in range(window, len(train_data)):
        x_train.append(train_data[i - window : i, 0])
        y_train.append(train_data[i, 0])

    x_train_arr = np.array(x_train)
    y_train_arr = np.array(y_train)
    # Reshape → (samples, timesteps, features)
    x_train_arr = x_train_arr.reshape(x_train_arr.shape[0], x_train_arr.shape[1], 1)

    # ── Test sequences ────────────────────────────────────────────────────────
    test_data = scaled_data[training_data_len - window :, :]
    x_test: list = []
    y_test: np.ndarray = dataset[training_data_len:, :]

    for i in range(window, len(test_data)):
        x_test.append(test_data[i - window : i, 0])

    x_test_arr = np.array(x_test)
    x_test_arr = x_test_arr.reshape(x_test_arr.shape[0], x_test_arr.shape[1], 1)

    return x_train_arr, y_train_arr, x_test_arr, y_test, scaler, training_data_len, scaled_data


def build_future_sequence(
    scaled_data: np.ndarray,
    model,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, list]:
    """
    Autoregressively forecast `config.NUM_FUTURE_DAYS` steps beyond the
    last known date.

    Parameters
    ----------
    scaled_data : full scaled closing-price array
    model       : trained model with a .predict() method
    scaler      : fitted MinMaxScaler

    Returns
    -------
    future_predictions_usd : array of shape (NUM_FUTURE_DAYS, 1)
    future_dates           : list of datetime objects for the x-axis
    """
    from datetime import timedelta
    import pandas as pd

    current_sequence = scaled_data[-config.LOOKBACK_WINDOW :].copy()
    future_preds_scaled: list = []

    print(f"Generating {config.NUM_FUTURE_DAYS} future predictions…")
    for _ in range(config.NUM_FUTURE_DAYS):
        seq = current_sequence.reshape(1, current_sequence.shape[0], 1)
        scaled_pred = model.predict(seq, verbose=0)
        future_preds_scaled.append(scaled_pred[0][0])
        current_sequence = np.append(current_sequence[1:], scaled_pred, axis=0)

    future_predictions_usd: np.ndarray = scaler.inverse_transform(
        np.array(future_preds_scaled).reshape(-1, 1)
    )
    print("Future prediction generation complete.")

    # Build a simple calendar list (includes weekends — filter downstream if desired)
    last_known_date = pd.Timestamp.now().normalize()
    future_dates = [
        last_known_date + timedelta(days=x)
        for x in range(1, config.NUM_FUTURE_DAYS + 1)
    ]

    return future_predictions_usd, future_dates
