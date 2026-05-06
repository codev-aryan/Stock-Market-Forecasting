"""
models/rf_model.py
------------------
Random Forest baseline implementation of BaseModel.

Key design note
---------------
The shared data pipeline outputs 3D arrays of shape (samples, 60, 1) to
satisfy Keras/LSTM expectations.  scikit-learn's RandomForestRegressor
requires 2D input (samples, features), so this class handles the reshape
internally — callers never need to think about it.

  train(x_train_3d, y_train)  →  flattens to (samples, 60) before fitting
  predict(x_test_3d)          →  flattens, predicts, reshapes to (samples, 1)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel
import config


class RFModel(BaseModel):
    """
    Random Forest regressor wrapped in the BaseModel interface.

    Usage
    -----
    >>> rf = RFModel()
    >>> rf.train(x_train, y_train)
    >>> scaled_preds = rf.predict(x_test)          # shape (samples, 1)
    >>> predictions  = scaler.inverse_transform(scaled_preds)
    """

    def __init__(
        self,
        n_estimators: int = config.RF_N_ESTIMATORS,
        random_state: int = config.RF_RANDOM_STATE,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,          # use all available cores
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _flatten(x: np.ndarray) -> np.ndarray:
        """
        Collapse 3D pipeline output → 2D sklearn input.

        (samples, timesteps, 1)  →  (samples, timesteps)
        2D arrays are passed through unchanged so the method is safe to
        call unconditionally.
        """
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        return x

    # ── BaseModel interface ───────────────────────────────────────────────────

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fit the Random Forest on the (flattened) training data.

        Parameters
        ----------
        x_train : shape (samples, 60, 1)  — as produced by data_pipeline
        y_train : shape (samples,)
        """
        print(
            f"Training RandomForestRegressor "
            f"(n_estimators={self._model.n_estimators}) …"
        )
        self._model.fit(self._flatten(x_train), y_train, **kwargs)
        print("Random Forest training complete.")

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return scaled predictions with shape (samples, 1) so that
        MinMaxScaler.inverse_transform() works without any extra reshaping
        by the caller — identical contract to LSTMModel.predict().

        Parameters
        ----------
        x : shape (samples, 60, 1)
        """
        preds = self._model.predict(self._flatten(x), **kwargs)
        return preds.reshape(-1, 1)

    # ── Convenience passthrough ───────────────────────────────────────────────

    @property
    def feature_importances(self) -> np.ndarray:
        """
        Relative importance of each of the 60 lag features.
        Useful for a quick sanity-check plot after training.
        """
        return self._model.feature_importances_

    @property
    def sklearn_model(self) -> RandomForestRegressor:
        """Direct access to the underlying sklearn estimator if needed."""
        return self._model
