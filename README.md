# Stock Price Prediction — LSTM vs Random Forest

Predicts Apple's stock closing price using a two-layer LSTM neural network and a Random Forest baseline. Both models are trained on the same 60-day lookback sequences and evaluated side by side using RMSE. A 60-day future forecast is generated from the trained LSTM.

---

## Project Structure

```
.
├── LICENSE
├── README.md
├── notebooks
│   ├── Prediction_clean.ipynb     # clean notebook (no output)
│   └── Prediction_output.ipynb   # notebook with saved outputs
├── requirements.txt
└── src
    ├── config.py                  # all hyperparameters
    ├── data_pipeline.py           # data download, scaling, sequencing
    ├── main.py                    # orchestrator
    ├── visualizer.py              # all plots
    └── models
        ├── base_model.py          # abstract base class
        ├── lstm_model.py          # LSTM model
        └── rf_model.py            # Random Forest model
```

---

## Models

### LSTM
Two-layer LSTM with dropout regularisation and early stopping.

```
LSTM(128, return_sequences=True,  dropout=0.2, recurrent_dropout=0.2)
LSTM(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
Dense(25)
Dense(1)
```

Trained with `Adam` optimizer, `MSE` loss, `EarlyStopping(patience=10)` on `val_loss`, and a 10% validation split.

### Random Forest
`RandomForestRegressor(n_estimators=100)` used as a traditional ML baseline. The 3D input `(samples, 60, 1)` is flattened to `(samples, 60)` internally before fitting, so both models share the exact same data pipeline.

---

## Key Design Decisions

**Leakage-free scaling** — `MinMaxScaler` is fit only on training data, then applied to the full dataset. The original approach (`fit_transform` on all data) would let the scaler see future prices during training.

**85/15 train-test split** — gives a larger test window (~500 days) for more reliable evaluation compared to the original 95/5 split.

**60-day lookback** — the model sees the past 60 trading days to predict the next day's close price.

**Business-day forecast** — future dates are generated with `pd.bdate_range`, skipping weekends.

---

## Hyperparameters

All values live in `src/config.py`.

| Parameter | Value |
|---|---|
| Lookback window | 60 days |
| Train / test split | 85% / 15% |
| Epochs (max) | 50 |
| Batch size | 32 |
| Dropout | 0.2 |
| Recurrent dropout | 0.2 |
| Early stopping patience | 10 |
| Validation split | 10% |
| Optimizer | Adam |
| Loss | MSE |
| RF estimators | 100 |
| RF random state | 42 |
| Future forecast days | 60 |

---

## Setup

```bash
git clone https://github.com/codev-aryan/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

> `keras` is bundled inside `tensorflow` — no separate install needed.

---

## Usage

### Run as a script

```bash
cd src
python main.py
```

### Run as a notebook

Open `notebooks/Prediction_clean.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

---

## Pipeline

```
Download EDA data (AAPL, GOOG, MSFT, AMZN — last 1 year)
        ↓
EDA plots: closing prices, volume, moving averages,
           daily returns, correlation heatmaps
        ↓
Download AAPL data (2012 → present)
        ↓
Scale (train only) → build 60-day sequences → train/test split
        ↓
Train LSTM  ←→  Train Random Forest
        ↓
Evaluate both — RMSE on test set
        ↓
Plot predictions (LSTM vs RF vs actual)
        ↓
RF feature importances (which lag day matters most)
        ↓
LSTM autoregressive 60-day future forecast
```

---

## EDA

Covers the last 1 year of data across all four tickers:

- Closing price history
- Trading volume
- Moving averages — 10, 20, 50 days
- Daily return percentage
- Pairplot of return correlations
- Heatmaps — return correlation and closing price correlation

---

## Evaluation

Both models are evaluated on the same held-out test set (last 15% of AAPL data, ~2012–present). Metric used is **Root Mean Squared Error (RMSE)** in USD.

The final chart overlays training data, validation actuals, LSTM predictions, RF predictions, and the 60-day future forecast on a single plot.
