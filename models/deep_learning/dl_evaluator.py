"""
models/deep_learning/dl_evaluator.py
──────────────────────────────────────
Generic DLEvaluator — timed test-set evaluation and artifact saving
for any DLTrainer-backed model.

Wraps models/shared/metrics.py in the same way as ClassicalEvaluator,
so the comparative harness can treat all tiers uniformly.

Usage:
    evaluator = DLEvaluator(trainer, model_name="CNN", results_dir=CNN_RESULTS_DIR)
    metrics, y_pred, y_prob = evaluator.evaluate(test_loader, y_true, train_time_s)
    evaluator.save_artifacts(y_true, y_pred, y_prob, metrics)
"""

import time
import logging
import pathlib

import numpy as np
from torch.utils.data import DataLoader

from models.deep_learning.dl_trainer import DLTrainer
from models.shared.metrics import (
    compute_metrics,
    add_timing,
    save_all_evaluation_artifacts,
)

logger = logging.getLogger(__name__)


class DLEvaluator:
    """
    Generic evaluator for any DLTrainer-backed model.
    Identical interface to LSTMEvaluator and ClassicalEvaluator.
    """

    def __init__(
        self,
        trainer:     DLTrainer,
        model_name:  str,
        results_dir: pathlib.Path,
    ):
        self.trainer     = trainer
        self.name        = model_name
        self.results_dir = results_dir

    def evaluate(
        self,
        test_loader:  DataLoader,
        y_true:       list | np.ndarray,
        train_time_s: float = 0.0,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Timed inference on test set + full metric computation.

        Returns:
            (metrics dict, y_pred array, y_prob array of P(positive))
        """
        y_true = np.asarray(y_true)
        n      = len(y_true)

        t0     = time.perf_counter()
        y_pred = self.trainer.predict(test_loader)       # (N,) int
        proba  = self.trainer.predict_proba(test_loader) # (N, 2)
        infer_time = time.perf_counter() - t0

        y_prob = proba[:, 1]   # P(positive)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics = add_timing(metrics, train_time_s, infer_time, n)
        metrics["model"]  = self.name
        metrics["tier"]   = "deep_learning"
        metrics["n_test"] = n

        logger.info(
            "[%s] acc=%.4f  f1=%.4f  auc=%.4f  infer=%.3fms/sample",
            self.name,
            metrics["accuracy"], metrics["f1_macro"],
            metrics["roc_auc"],  metrics["infer_ms_per_sample"],
        )
        return metrics, y_pred, y_prob

    def save_artifacts(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        y_prob:     np.ndarray,
        metrics:    dict,
    ) -> pathlib.Path:
        """Save metrics.json, classification_report.txt, confusion_matrix.png."""
        save_all_evaluation_artifacts(
            y_true, y_pred, y_prob, metrics, self.results_dir, self.name,
        )
        logger.info("[%s] Artifacts saved -> %s", self.name, self.results_dir)
        return self.results_dir
