"""
models/classical/trainers.py
─────────────────────────────
Trainer classes for the three classical ML models.

Each trainer:
  - inherits BaseTrainer (unified interface)
  - wraps a sklearn estimator
  - enforces fit-on-train-only
  - provides predict() and predict_proba()
  - handles serialization to/from disk

SVM note: LinearSVC does not natively support predict_proba.
We wrap it with CalibratedClassifierCV(method='sigmoid') which
fits a Platt scaling calibration on held-out folds.
"""

import time
import pathlib
import logging

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from models.base import BaseTrainer
from models.classical.config import (
    NAIVE_BAYES_CONFIG, LOGISTIC_REGRESSION_CONFIG, SVM_CONFIG,
    saved_model_dir, DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)

_MODEL_FILE = "model.joblib"


# ── Base classical trainer ────────────────────────────────────────────────────

class BaseClassicalTrainer(BaseTrainer):
    """
    Shared implementation for all classical ML trainers.
    Subclasses only need to define self.model in __init__.
    """

    tier: str = "classical"

    def train(self, X_train: spmatrix, y_train: list | np.ndarray) -> float:
        """
        Fit the model on training data.
        Returns wall-clock training time in seconds.
        ⚠️  Never call with test data.
        """
        logger.info("[%s] Training on %d samples …", self.name, X_train.shape[0])
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        self._trained = True
        logger.info("[%s] Training done in %.2fs", self.name, elapsed)
        return elapsed

    def predict(self, X: spmatrix) -> np.ndarray:
        self._assert_trained()
        return self.model.predict(X)

    def predict_proba(self, X: spmatrix) -> np.ndarray:
        """
        Return probability array of shape (N, 2).
        Column 0 = P(negative), Column 1 = P(positive).
        """
        self._assert_trained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Decision function fallback (should not reach here in normal usage)
        scores = self.model.decision_function(X)
        proba  = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - proba, proba])

    def save(self, directory: pathlib.Path | None = None) -> pathlib.Path:
        self._assert_trained()
        out_dir = directory or saved_model_dir(self.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / _MODEL_FILE
        joblib.dump(self.model, path)
        logger.info("[%s] Model saved → %s", self.name, path)
        return path

    @classmethod
    def load(cls, directory: pathlib.Path | None = None) -> "BaseClassicalTrainer":
        key     = DISPLAY_NAMES.get(cls._model_key, cls._model_key)
        out_dir = directory or saved_model_dir(key)
        path    = out_dir / _MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at {path}. Run run_training.py first."
            )
        instance         = cls.__new__(cls)
        instance.name    = key
        instance.model   = joblib.load(path)
        instance._trained = True
        logger.info("[%s] Model loaded ← %s", key, path)
        return instance

    def _assert_trained(self) -> None:
        if not getattr(self, "_trained", False):
            raise RuntimeError(
                f"{self.name} is not trained. Call train(X_train, y_train) first."
            )


# ── Naive Bayes ────────────────────────────────────────────────────────────────

class NaiveBayesTrainer(BaseClassicalTrainer):
    """
    Multinomial Naive Bayes.

    Requires non-negative TF-IDF features (ensured by sublinear_tf=True).
    Native predict_proba via log-probabilities.
    """

    _model_key = "naive_bayes"

    def __init__(self, config: dict | None = None):
        cfg        = config or NAIVE_BAYES_CONFIG
        self.name  = DISPLAY_NAMES["naive_bayes"]
        self.model = MultinomialNB(**cfg)
        self._trained = False


# ── Logistic Regression ────────────────────────────────────────────────────────

class LogisticRegressionTrainer(BaseClassicalTrainer):
    """
    Logistic Regression with L2 regularization.
    lbfgs solver — efficient on dense feature sets up to ~100k features.
    Native predict_proba via sigmoid activation.
    """

    _model_key = "logistic_regression"

    def __init__(self, config: dict | None = None):
        cfg        = config or LOGISTIC_REGRESSION_CONFIG
        self.name  = DISPLAY_NAMES["logistic_regression"]
        self.model = LogisticRegression(**cfg)
        self._trained = False


# ── SVM ───────────────────────────────────────────────────────────────────────

class SVMTrainer(BaseClassicalTrainer):
    """
    Linear SVM wrapped with Platt scaling calibration for predict_proba.

    LinearSVC is significantly faster than SVC(kernel='linear') at the cost of
    not natively supporting probabilities. CalibratedClassifierCV adds Platt
    scaling via 3-fold cross-validation on the training data.

    This is the standard approach for production SVM deployments that need
    calibrated confidence scores.
    """

    _model_key = "svm"

    def __init__(self, config: dict | None = None):
        cfg   = config or SVM_CONFIG
        base  = LinearSVC(C=cfg["svc_C"], max_iter=cfg["svc_max_iter"], random_state=42)
        self.name  = DISPLAY_NAMES["svm"]
        self.model = CalibratedClassifierCV(
            base,
            cv=cfg["calibration_cv"],
            method=cfg["calibration_method"],
        )
        self._trained = False
