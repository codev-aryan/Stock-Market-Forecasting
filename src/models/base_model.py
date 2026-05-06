"""
models/base_model.py
--------------------
Abstract Base Class that every model in this project must inherit from.
Any future model (e.g. RandomForest, XGBoost) must implement:
  - train()
  - predict()
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Common interface for all stock-prediction models.

    Concrete subclasses must override `train` and `predict`.
    Optional: override `evaluate` to return custom metrics.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Fit the model on training data.

        Parameters
        ----------
        x_train : feature array of shape (samples, timesteps, features)
                  for sequential models, or (samples, features) for flat models
        y_train : target array of shape (samples,)
        **kwargs: additional keyword arguments passed to the underlying fit call
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for input `x`.

        Parameters
        ----------
        x       : feature array — same shape contract as x_train
        **kwargs: additional keyword arguments

        Returns
        -------
        np.ndarray of predictions in the *scaled* space.
        Inverse-transform with the pipeline's scaler before plotting.
        """
        ...

    def evaluate(self, predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """
        Compute evaluation metrics.  Default: RMSE only.
        Override in subclasses to add more metrics.

        Returns
        -------
        dict with at least the key 'rmse'
        """
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        return {"rmse": rmse}
