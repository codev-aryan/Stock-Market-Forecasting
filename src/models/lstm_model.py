"""
models/lstm_model.py
--------------------
LSTM implementation of BaseModel.

Fixes applied (v2)
------------------
1. EPOCHS      : 1  -> 50   (model needs multiple passes to learn trends)
2. BATCH_SIZE  : 1  -> 32   (stable gradient estimates; far faster training)
3. Dropout     : added after both LSTM layers to prevent overfitting on noisy
                 stock data (rate configurable via config.DROPOUT_RATE)
4. EarlyStopping: training halts automatically when val_loss stops improving,
                  avoiding wasted compute and late-stage overfitting

Updated Architecture
  LSTM(128, return_sequences=True,  dropout=0.2, recurrent_dropout=0.2)
  LSTM(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
  Dense(25)
  Dense(1)
"""

from __future__ import annotations

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

from models.base_model import BaseModel
import config


class LSTMModel(BaseModel):
    """
    Two-layer LSTM with Dropout and EarlyStopping for univariate
    time-series forecasting.
    """

    def __init__(self) -> None:
        self._model: Sequential = self._build()
        self.history = None   # stores Keras History object after training

    def _build(self) -> Sequential:
        """Construct and compile the Keras model."""
        model = Sequential([
            # FIX 3 - Dropout on LSTM layers prevents memorising training noise
            LSTM(
                config.LSTM_UNITS_LAYER_1,
                return_sequences=True,
                input_shape=(config.LOOKBACK_WINDOW, 1),
                dropout=config.DROPOUT_RATE,
                recurrent_dropout=config.RECURRENT_DROPOUT_RATE,
            ),
            LSTM(
                config.LSTM_UNITS_LAYER_2,
                return_sequences=False,
                dropout=config.DROPOUT_RATE,
                recurrent_dropout=config.RECURRENT_DROPOUT_RATE,
            ),
            Dense(config.DENSE_UNITS),
            Dense(1),
        ])
        model.compile(optimizer=config.OPTIMIZER, loss=config.LOSS)
        model.summary()
        return model

    def train(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        
        epochs     = kwargs.pop("epochs",     config.EPOCHS)
        batch_size = kwargs.pop("batch_size", config.BATCH_SIZE)

        early_stop = EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        user_callbacks = kwargs.pop("callbacks", [])
        all_callbacks  = [early_stop] + list(user_callbacks)

        self.history = self._model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=all_callbacks,
            **kwargs,
        )

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._model.predict(x, **kwargs)

    @property
    def keras_model(self) -> Sequential:
        return self._model

    @property
    def training_history(self):
        if self.history is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.history
