"""
models/base.py
───────────────
Abstract base classes that every model tier (Classical ML, LSTM, CNN-LSTM,
BERT, DistilBERT) must implement.

Defines the unified interface used by the comparative inference system.
"""

from abc import ABC, abstractmethod
from typing import Any
import pathlib
import numpy as np


class BaseTrainer(ABC):
    """
    Common contract for all model trainers across all tiers.
    Training scripts subclass this; the run_* orchestrators call the interface.
    """

    name: str           # Human-readable model name, e.g. "Logistic Regression"
    tier: str           # "classical" | "deep_learning" | "transformer"

    @abstractmethod
    def train(self, *args, **kwargs) -> float:
        """Train the model. Returns wall-clock training time in seconds."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return integer class predictions for input X."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return probability array of shape (N, num_classes)."""

    @abstractmethod
    def save(self, directory: pathlib.Path) -> pathlib.Path:
        """Serialize model to directory. Returns path written."""

    @classmethod
    @abstractmethod
    def load(cls, directory: pathlib.Path) -> "BaseTrainer":
        """Restore a previously saved model from directory."""


class BasePredictor(ABC):
    """
    Unified inference interface for the comparative system.
    Each tier implements this so the comparison harness can call
    predict() / predict_single() identically across all 7 models.
    """

    model_name: str     # Short identifier, e.g. "svm"
    tier: str

    @abstractmethod
    def predict(self, texts: list[str]) -> dict:
        """
        Run inference on a list of raw review strings.

        Returns:
            {
              "labels":      list[int],        # 0 = negative, 1 = positive
              "label_names": list[str],
              "confidences": list[float],      # P(positive)
              "latency_ms":  float,            # total batch latency
              "ms_per_sample": float,
            }
        """

    @abstractmethod
    def predict_single(self, text: str) -> dict:
        """
        Run inference on a single raw review string.

        Returns:
            {
              "label":      int,
              "label_name": str,
              "confidence": float,   # P(positive)
              "latency_ms": float,
            }
        """
