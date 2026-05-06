"""
config.py
---------
Central configuration for the stock price prediction project.
All tunable hyperparameters and constants live here.
"""

from datetime import datetime

# ── Tickers ───────────────────────────────────────────────────────────────────
TECH_LIST: list[str] = ["AAPL", "GOOG", "MSFT", "AMZN"]
COMPANY_NAMES: list[str] = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

# Primary ticker used for the LSTM prediction
PRIMARY_TICKER: str = "AAPL"
PRIMARY_TICKER_NAME: str = "Apple"

# ── Date Ranges ───────────────────────────────────────────────────────────────
# EDA date range  → last 1 year
EDA_END_DATE: datetime = datetime.now()
EDA_START_DATE: datetime = datetime(
    EDA_END_DATE.year - 1, EDA_END_DATE.month, EDA_END_DATE.day
)

# LSTM training date range
LSTM_START_DATE: str = "2012-01-01"
LSTM_END_DATE: datetime = datetime.now()

# ── Data Pipeline ─────────────────────────────────────────────────────────────
LOOKBACK_WINDOW: int = 60          # Number of past days used as features
TRAIN_SPLIT_RATIO: float = 0.95    # Fraction of data used for training
MOVING_AVG_WINDOWS: list[int] = [10, 20, 50]  # MA periods to compute

# ── Model Hyperparameters ─────────────────────────────────────────────────────
LSTM_UNITS_LAYER_1: int = 128
LSTM_UNITS_LAYER_2: int = 64
DENSE_UNITS: int = 25
EPOCHS: int = 1
BATCH_SIZE: int = 1
OPTIMIZER: str = "adam"
LOSS: str = "mean_squared_error"

# ── Random Forest Hyperparameters ─────────────────────────────────────────────
RF_N_ESTIMATORS: int = 100   # Number of trees in the forest
RF_RANDOM_STATE: int = 42    # Seed for reproducibility

# ── Forecasting ───────────────────────────────────────────────────────────────
NUM_FUTURE_DAYS: int = 60          # How many calendar days to forecast ahead
