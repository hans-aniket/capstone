"""
models/transformers/evaluator.py
──────────────────────────────────
TransformerEvaluator — timed test-set evaluation and artifact saving
for any TransformerTrainer-backed model.

Identical in concept to DLEvaluator and ClassicalEvaluator.
"""

import time
import logging
import pathlib
import numpy as np
from torch.utils.data import DataLoader

from models.transformers.trainer import TransformerTrainer
from models.shared.metrics import compute_metrics, add_timing, save_all_evaluation_artifacts

logger = logging.getLogger(__name__)

class TransformerEvaluator:
    def __init__(
        self,
        trainer:     TransformerTrainer,
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
        y_true = np.asarray(y_true)
        n      = len(y_true)

        t0     = time.perf_counter()
        y_pred = self.trainer.predict(test_loader)
        proba  = self.trainer.predict_proba(test_loader)
        infer_time = time.perf_counter() - t0

        y_prob = proba[:, 1]

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics = add_timing(metrics, train_time_s, infer_time, n)
        metrics["model"]  = self.name
        metrics["tier"]   = "transformers"
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
        save_all_evaluation_artifacts(
            y_true, y_pred, y_prob, metrics, self.results_dir, self.name,
        )
        logger.info("[%s] Artifacts saved -> %s", self.name, self.results_dir)
        return self.results_dir
