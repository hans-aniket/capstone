"""
models/classical/predictor.py
──────────────────────────────
Unified inference interface for all three classical ML models.

Implements BasePredictor so the comparative harness can call
predict() identically across Classical ML, LSTM, CNN-LSTM, BERT, DistilBERT.

Usage:
    # After training:
    predictor = ClassicalPredictor.from_saved("svm")
    result    = predictor.predict_single("Terrible product, broke in a week.")
    results   = predictor.predict(["Great value!", "Stopped working after a day."])
"""

import time
import pathlib
import logging

import numpy as np

from models.base import BasePredictor
from models.classical.trainers import (
    NaiveBayesTrainer,
    LogisticRegressionTrainer,
    SVMTrainer,
    BaseClassicalTrainer,
)
from models.classical.config import saved_model_dir, DISPLAY_NAMES
from preprocessing.classical_preprocessor import ClassicalPreprocessor
from preprocessing.config import LABEL_MAP

logger = logging.getLogger(__name__)

# Maps short key → trainer class
_TRAINER_MAP: dict[str, type[BaseClassicalTrainer]] = {
    "naive_bayes":         NaiveBayesTrainer,
    "logistic_regression": LogisticRegressionTrainer,
    "svm":                 SVMTrainer,
}


class ClassicalPredictor(BasePredictor):
    """
    Loads a saved classical ML model + the fitted TF-IDF vectorizer
    and exposes a clean predict / predict_single interface.

    This is the object that the comparative system calls at inference time.
    It is intentionally model-agnostic at the call site — all three classical
    models expose the same methods with the same return schema.
    """

    tier: str = "classical"

    def __init__(
        self,
        model_name:   str,
        trainer:      BaseClassicalTrainer,
        preprocessor: ClassicalPreprocessor,
    ):
        self.model_name   = model_name
        self.trainer      = trainer
        self.preprocessor = preprocessor
        logger.info("ClassicalPredictor ready: %s", DISPLAY_NAMES.get(model_name, model_name))

    # ── Factory: load from disk ────────────────────────────────────────────────

    @classmethod
    def from_saved(cls, model_name: str) -> "ClassicalPredictor":
        """
        Load a trained model and the fitted TF-IDF vectorizer from disk.

        Args:
            model_name: One of "naive_bayes", "logistic_regression", "svm".
        """
        if model_name not in _TRAINER_MAP:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_TRAINER_MAP.keys())}"
            )
        trainer_cls = _TRAINER_MAP[model_name]
        trainer     = trainer_cls.load(saved_model_dir(DISPLAY_NAMES[model_name]))
        preprocessor = ClassicalPreprocessor.load()
        return cls(model_name, trainer, preprocessor)

    # ── Core inference ─────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> dict:
        """
        Run batch inference on raw review strings.

        Returns:
            {
              "labels":        list[int],       0=negative, 1=positive
              "label_names":   list[str],
              "confidences":   list[float],     P(positive)
              "latency_ms":    float,            total batch time
              "ms_per_sample": float,
            }
        """
        t0      = time.perf_counter()
        X       = self.preprocessor.transform(texts)
        labels  = self.trainer.predict(X).tolist()
        proba   = self.trainer.predict_proba(X)[:, 1].tolist()
        elapsed = (time.perf_counter() - t0) * 1000   # ms

        return {
            "labels":        labels,
            "label_names":   [LABEL_MAP[l] for l in labels],
            "confidences":   [round(p, 4) for p in proba],
            "latency_ms":    round(elapsed, 3),
            "ms_per_sample": round(elapsed / max(len(texts), 1), 4),
        }

    def predict_single(self, text: str) -> dict:
        """
        Run inference on a single raw review string.

        Returns:
            {
              "label":      int,
              "label_name": str,
              "confidence": float,   P(positive)
              "latency_ms": float,
            }
        """
        result = self.predict([text])
        return {
            "label":      result["labels"][0],
            "label_name": result["label_names"][0],
            "confidence": result["confidences"][0],
            "latency_ms": result["latency_ms"],
        }

    def __repr__(self) -> str:
        display = DISPLAY_NAMES.get(self.model_name, self.model_name)
        return f"ClassicalPredictor(model='{display}', tier='classical')"
