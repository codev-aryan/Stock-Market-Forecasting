"""
models/lstm_model.py
--------------------
LSTM implementation of BaseModel using the exact architecture from
the original Prediction.py notebook.

Architecture
  LSTM(128, return_sequences=True)
  LSTM(64,  return_sequences=False)
  Dense(25)
  Dense(1)
"""

from __future__ import annotations

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

from models.base_model import BaseModel
import config


class LSTMModel(BaseModel):
    """
    Two-layer LSTM for univariate time-series forecasting.

    Usage
    -----
    >>> model = LSTMModel()
    >>> model.train(x_train, y_train)
    >>> scaled_preds = model.predict(x_test)
    """

    def __init__(self) -> None:
        self._model: Sequential = self._build()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build(self) -> Sequential:
        """Construct and compile the Keras model."""
        model = Sequential(
            [
                LSTM(
                    config.LSTM_UNITS_LAYER_1,
                    return_sequences=True,
                    input_shape=(config.LOOKBACK_WINDOW, 1),
                ),
                LSTM(config.LSTM_UNITS_LAYER_2, return_sequences=False),
                Dense(config.DENSE_UNITS),
                Dense(1),
            ]
        )
        model.compile(optimizer=config.OPTIMIZER, loss=config.LOSS)
        model.summary()
        return model

    # ── BaseModel interface ───────────────────────────────────────────────────

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fit the LSTM.

        Epoch count and batch size are taken from config.py unless
        overridden via kwargs (epochs=, batch_size=).
        """
        epochs = kwargs.pop("epochs", config.EPOCHS)
        batch_size = kwargs.pop("batch_size", config.BATCH_SIZE)

        self._model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return raw scaled predictions.

        Use `data_pipeline.build_future_sequence` for autoregressive
        multi-step forecasting.
        """
        return self._model.predict(x, **kwargs)

    # ── Convenience passthrough ───────────────────────────────────────────────

    @property
    def keras_model(self) -> Sequential:
        """Direct access to the underlying Keras model if needed."""
        return self._model
