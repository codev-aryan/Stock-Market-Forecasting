<div align="center">

# 📈 Stock Price Prediction
### LSTM Neural Network vs Random Forest

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  End-to-end stock price forecasting pipeline that trains a two-layer LSTM and a Random Forest baseline on Apple's historical prices, evaluates both against the same test set, and generates a 60-day future forecast.
</p>

</div>

---

## Overview

This project builds a complete forecasting pipeline from raw data to future predictions, with a focus on doing things correctly — no data leakage, proper train/test split, regularised model, and a traditional ML baseline to put the deep learning results in context.

**Tickers covered in EDA:** `AAPL` · `GOOG` · `MSFT` · `AMZN`  
**Prediction target:** Apple (`AAPL`) closing price  
**Training data:** January 2012 → present  
**Evaluation metric:** RMSE (USD)

---

## Features

- **Two models trained on identical data** — LSTM and Random Forest share the same 60-day lookback sequences so the comparison is fair
- **Leakage-free scaling** — `MinMaxScaler` is fit only on training data before transforming the full dataset
- **Regularised LSTM** — dropout + recurrent dropout on both layers, with early stopping that restores the best weights
- **60-day autoregressive forecast** — each LSTM prediction feeds back as input for the next step, on business days only
- **OOP architecture** — `BaseModel` ABC enforces a consistent `train()` / `predict()` / `evaluate()` interface across all models
- **Fully modular** — swap a model, change a hyperparameter, or add a new ticker without touching anything else

---

## Project Structure

```
.
├── notebooks
│   ├── Prediction_clean.ipynb     ← run this (no pre-saved output)
│   └── Prediction_output.ipynb   ← pre-run version with all outputs
├── src
│   ├── config.py                  ← every hyperparameter in one place
│   ├── data_pipeline.py           ← download · scale · sequence · split
│   ├── visualizer.py              ← all matplotlib / seaborn plots
│   ├── main.py                    ← orchestrator (runs the full pipeline)
│   └── models
│       ├── base_model.py          ← abstract base class (ABC)
│       ├── lstm_model.py          ← LSTM with dropout + early stopping
│       └── rf_model.py            ← Random Forest baseline
├── requirements.txt
└── LICENSE
```

---

## Architecture

### LSTM Model

```
Input → (samples, 60, 1)
        │
        ▼
LSTM(128, return_sequences=True,  dropout=0.2, recurrent_dropout=0.2)
        │
        ▼
LSTM(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
        │
        ▼
Dense(25)
        │
        ▼
Dense(1)  →  predicted next-day close price (scaled)
```

Compiled with `Adam` · `MSE loss` · `EarlyStopping(patience=10, restore_best_weights=True)`

### Random Forest Baseline

`RandomForestRegressor(n_estimators=100)` trained on the same sequences. The 3D array `(samples, 60, 1)` is flattened to `(samples, 60)` inside the model class — callers use the same API as the LSTM.

### Abstract Base Class

Every model inherits from `BaseModel` (ABC) and must implement:

```python
def train(self, x_train, y_train, **kwargs) -> None
def predict(self, x, **kwargs) -> np.ndarray
```

`evaluate()` is provided by default (returns RMSE dict) and can be overridden.

---

## Pipeline

```
1.  Download 1-year data  →  AAPL · GOOG · MSFT · AMZN
2.  EDA  →  closing prices · volume · MAs · daily returns · correlation
           ─────────────────────────────────────────────────────────
3.  Download AAPL  2012 → present
4.  Fit scaler on train split only  (no leakage)
5.  Build 60-day lookback sequences
6.  85 / 15 train-test split
           ─────────────────────────────────────────────────────────
7.  Train LSTM          (epochs=50, batch=32, early stopping)
8.  Train Random Forest (n_estimators=100)
           ─────────────────────────────────────────────────────────
9.  Evaluate both  →  RMSE on held-out test set
10. Plot  →  actuals · LSTM predictions · RF predictions
11. RF feature importances  →  which of the 60 lag days matters most
12. LSTM autoregressive forecast  →  60 business days ahead
```

---

## Hyperparameters

All values are in `src/config.py` — one place to change anything.

| Parameter | Value |
|:---|:---|
| Lookback window | 60 days |
| Train / test split | 85% / 15% |
| Epochs (max) | 50 |
| Batch size | 32 |
| Dropout rate | 0.2 |
| Recurrent dropout | 0.2 |
| Early stopping patience | 10 |
| Validation split | 10% |
| Optimizer | Adam |
| Loss | Mean Squared Error |
| RF n\_estimators | 100 |
| RF random\_state | 42 |
| Future forecast days | 60 |

---

## Getting Started

**Clone and install**

```bash
git clone https://github.com/codev-aryan/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

**Run the notebook** *(recommended)*

```
notebooks/Prediction_clean.ipynb
```

Open in Jupyter Lab, Jupyter Notebook, or [Google Colab](https://colab.research.google.com/) and run all cells top to bottom.

**Run as a script**

```bash
cd src
python main.py
```

---

## Tech Stack

| Library | Purpose |
|:---|:---|
| `yfinance` | Download historical OHLCV data |
| `pandas` / `numpy` | Data manipulation and sequence building |
| `scikit-learn` | MinMaxScaler, RandomForestRegressor |
| `tensorflow` / `keras` | LSTM model definition and training |
| `matplotlib` / `seaborn` | All visualisations |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built by <a href="https://github.com/codev-aryan">codev-aryan</a></sub>
</div>
