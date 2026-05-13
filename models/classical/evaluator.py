"""
models/classical/evaluator.py
──────────────────────────────
Evaluation workflow for classical ML models.

Wraps models/shared/metrics.py with:
  - timed inference measurement
  - per-model output directory routing
  - full artifact saving in one call
"""

import time
import logging
import pathlib

import numpy as np
from scipy.sparse import spmatrix

from models.classical.config import model_dir, DISPLAY_NAMES
from models.classical.trainers import BaseClassicalTrainer
from models.shared.metrics import (
    compute_metrics,
    add_timing,
    save_all_evaluation_artifacts,
    generate_classification_report,
    LABEL_NAMES,
)

logger = logging.getLogger(__name__)


class ClassicalEvaluator:
    """
    Runs evaluation for a trained classical ML model and saves all artifacts.

    Usage:
        evaluator = ClassicalEvaluator(trainer)
        results   = evaluator.evaluate(X_test, y_test)
        evaluator.save_artifacts(y_true, y_pred, y_prob, results)
    """

    def __init__(self, trainer: BaseClassicalTrainer):
        self.trainer = trainer
        self.name    = trainer.name

    def evaluate(
        self,
        X:             spmatrix,
        y_true:        list | np.ndarray,
        train_time_s:  float = 0.0,
    ) -> dict:
        """
        Run timed inference and compute the full metric suite.

        Args:
            X:            Feature matrix (TF-IDF sparse, test split).
            y_true:       Ground-truth labels.
            train_time_s: Training time in seconds (pass-through for the report).

        Returns:
            Full metrics dict including timing.
        """
        y_true = np.asarray(y_true)
        n      = len(y_true)

        # ── Timed prediction ──────────────────────────────────────────────────
        t0     = time.perf_counter()
        y_pred = self.trainer.predict(X)
        y_prob = self.trainer.predict_proba(X)[:, 1]   # P(positive)
        infer_time = time.perf_counter() - t0

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics = add_timing(metrics, train_time_s, infer_time, n)
        metrics["model"]  = self.name
        metrics["tier"]   = "classical"
        metrics["n_test"] = n

        self._log_summary(metrics)
        return metrics, y_pred, y_prob

    def save_artifacts(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        y_prob:     np.ndarray,
        metrics:    dict,
        output_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Save metrics.json, classification_report.txt, confusion_matrix.png.
        Returns the output directory path.
        """
        out = output_dir or model_dir(self.name)
        save_all_evaluation_artifacts(
            y_true, y_pred, y_prob, metrics, out, self.name,
        )
        logger.info("[%s] Artifacts saved → %s", self.name, out)
        return out

    def _log_summary(self, metrics: dict) -> None:
        logger.info(
            "[%s] acc=%.4f  f1=%.4f  auc=%.4f  infer=%.3fms/sample",
            self.name,
            metrics.get("accuracy", 0),
            metrics.get("f1_macro", 0),
            metrics.get("roc_auc", 0),
            metrics.get("infer_ms_per_sample", 0),
        )
