"""
models/deep_learning/evaluator.py
───────────────────────────────────
LSTMEvaluator: timed test-set evaluation and artifact saving.

Wraps models/shared/metrics.py — identical output format to
ClassicalEvaluator so the comparison harness treats them uniformly.
"""

import time
import logging
import pathlib

import numpy as np
from torch.utils.data import DataLoader

from models.deep_learning.trainer import LSTMTrainer
from models.deep_learning.config import LSTM_RESULTS_DIR
from models.shared.metrics import (
    compute_metrics,
    add_timing,
    save_all_evaluation_artifacts,
    LABEL_NAMES,
)

logger = logging.getLogger(__name__)


class LSTMEvaluator:
    """
    Test-set evaluation for the trained LSTM model.

    Usage:
        evaluator = LSTMEvaluator(trainer)
        metrics, y_pred, y_prob = evaluator.evaluate(test_loader, train_time_s)
        evaluator.save_artifacts(y_true, y_pred, y_prob, metrics)
    """

    def __init__(self, trainer: LSTMTrainer):
        self.trainer    = trainer
        self.name       = "LSTM"

    def evaluate(
        self,
        test_loader:   DataLoader,
        y_true:        list | np.ndarray,
        train_time_s:  float = 0.0,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Run timed inference on the test set and compute the full metric suite.

        Args:
            test_loader:  DataLoader for the test split.
            y_true:       Ground-truth labels (numpy array or list).
            train_time_s: Wall-clock training time (pass-through for the report).

        Returns:
            (metrics dict, y_pred array, y_prob array of P(positive))
        """
        y_true = np.asarray(y_true)
        n      = len(y_true)

        # ── Timed inference ───────────────────────────────────────────────────
        t0     = time.perf_counter()
        y_pred = self.trainer.predict(test_loader)          # (N,) int array
        proba  = self.trainer.predict_proba(test_loader)    # (N, 2) float array
        infer_time = time.perf_counter() - t0

        y_prob = proba[:, 1]   # P(positive)

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics = add_timing(metrics, train_time_s, infer_time, n)
        metrics["model"]  = self.name
        metrics["tier"]   = "deep_learning"
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
        """
        out = output_dir or LSTM_RESULTS_DIR
        save_all_evaluation_artifacts(y_true, y_pred, y_prob, metrics, out, self.name)
        logger.info("[LSTM] Evaluation artifacts saved -> %s", out)
        return out

    def _log_summary(self, metrics: dict) -> None:
        logger.info(
            "[LSTM] acc=%.4f  f1=%.4f  auc=%.4f  infer=%.3fms/sample",
            metrics.get("accuracy", 0),
            metrics.get("f1_macro", 0),
            metrics.get("roc_auc", 0),
            metrics.get("infer_ms_per_sample", 0),
        )
