"""
main.py
-------
Orchestrator — ties every module together without containing any
business logic itself.

Pipeline
  1. Download EDA data  →  visualise
  2. Download LSTM data →  preprocess
  3. Instantiate & train LSTMModel
  4. Evaluate on test set
  5. Generate future forecast
  6. Plot results
"""

import pandas as pd
import numpy as np

import config
import data_pipeline as dp
import visualizer as viz
from models.lstm_model import LSTMModel


def main() -> None:

    # ── 1. EDA ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Downloading EDA data …")
    print("=" * 60)

    company_list, closing_df = dp.download_eda_data()
    tech_rets = closing_df.pct_change()

    viz.plot_closing_prices(company_list)
    viz.plot_volumes(company_list)
    viz.plot_moving_averages(company_list)
    viz.plot_daily_returns(company_list)
    viz.plot_pairplot(tech_rets)
    viz.plot_correlation_heatmaps(tech_rets, closing_df)

    # ── 2. LSTM data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Downloading & preprocessing LSTM data …")
    print("=" * 60)

    df = dp.download_lstm_data()
    viz.plot_close_history(df)

    (
        x_train,
        y_train,
        x_test,
        y_test,
        scaler,
        training_data_len,
        scaled_data,
    ) = dp.build_lstm_datasets(df)

    print(f"\nTraining samples : {len(x_train)}")
    print(f"Test samples     : {len(x_test)}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Building & training LSTM model …")
    print("=" * 60)

    lstm = LSTMModel()
    lstm.train(x_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluating on test set …")
    print("=" * 60)

    scaled_preds = lstm.predict(x_test)
    predictions = scaler.inverse_transform(scaled_preds)

    metrics = lstm.evaluate(predictions, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}")

    # Attach predictions to the validation slice
    data = df[["Close"]]
    train_slice = data[: training_data_len]
    valid_slice = data[training_data_len:].copy()
    valid_slice["Predictions"] = predictions[: len(valid_slice)]

    # Quick validation-only plot
    viz.plot_predictions(train_slice, valid_slice, future_df=None)

    # ── 5. Future forecast ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 5 — Generating {config.NUM_FUTURE_DAYS}-day future forecast …")
    print("=" * 60)

    future_predictions_usd, future_dates = dp.build_future_sequence(
        scaled_data, lstm, scaler
    )
    future_df = pd.DataFrame(
        index=future_dates,
        data={"Future Predictions": future_predictions_usd.flatten()},
    )

    # ── 6. Final plot ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Plotting final results …")
    print("=" * 60)

    # Use the original full df for train / valid labels on x-axis
    train_full = df[: training_data_len]
    valid_full = df[training_data_len:].copy()
    valid_full["Predictions"] = predictions[: len(valid_full)]

    viz.plot_predictions(train_full, valid_full, future_df=future_df)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
