"""
models/classical/run_training.py
─────────────────────────────────
Trains all three classical ML models (NB, LR, SVM) end-to-end.

Run from project root:
    python models/classical/run_training.py

Pipeline:
    1. Load train/test Parquet splits
    2. Run classical preprocessing (TF-IDF — fit on train, apply to test)
    3. Train each model, measure wall-clock time
    4. Evaluate on test set, measure inference latency
    5. Save model artifacts, metrics JSON, report, confusion matrix
    6. Save cross-model comparison JSON
    7. Print ranked summary table

Outputs (results/classical/<model>/):
    model saved    →  models/classical/saved/<model>/model.joblib
    metrics        →  results/classical/<model>/metrics.json
    report         →  results/classical/<model>/classification_report.txt
    confusion mat  →  results/classical/<model>/confusion_matrix.png
    comparison     →  results/classical/comparison.json
"""

import sys
import json
import logging
import pathlib
import random

import numpy as np
import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import SPLITS_DIR, SEED
from preprocessing.classical_preprocessor import ClassicalPreprocessor
from models.classical.config import RESULTS_DIR, DISPLAY_NAMES
from models.classical.trainers import NaiveBayesTrainer, LogisticRegressionTrainer, SVMTrainer
from models.classical.evaluator import ClassicalEvaluator
from models.shared.metrics import save_metrics_json

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classical_training")

# ── Trainers in run order ──────────────────────────────────────────────────────
TRAINERS = [
    NaiveBayesTrainer(),
    LogisticRegressionTrainer(),
    SVMTrainer(),
]


def lock_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = SPLITS_DIR / "train.parquet"
    test_path  = SPLITS_DIR / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Splits not found in {SPLITS_DIR}.\n"
            "Run:  python data/prepare_data.py  first."
        )
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    logger.info("Splits loaded | train: %d  test: %d", len(train_df), len(test_df))
    return train_df, test_df


def run_classical_pipeline() -> list[dict]:
    """
    Full train → evaluate → save pipeline for all classical models.
    Returns a list of per-model metric dicts.
    """
    lock_seeds()

    train_df, test_df = load_splits()
    train_texts  = train_df["text"].tolist()
    test_texts   = test_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # ── Preprocessing (shared TF-IDF, fit on train only) ─────────────────────
    logger.info("=" * 55)
    logger.info("Building TF-IDF representation …")
    preprocessor = ClassicalPreprocessor()
    X_train = preprocessor.fit_transform(train_texts)
    X_test  = preprocessor.transform(test_texts)
    preprocessor.save()
    logger.info("TF-IDF | train: %s  test: %s", X_train.shape, X_test.shape)

    all_metrics = []

    for trainer in TRAINERS:
        logger.info("=" * 55)
        logger.info("MODEL: %s", trainer.name)

        # ── Train ─────────────────────────────────────────────────────────────
        train_time = trainer.train(X_train, train_labels)

        # ── Evaluate ──────────────────────────────────────────────────────────
        evaluator = ClassicalEvaluator(trainer)
        metrics, y_pred, y_prob = evaluator.evaluate(
            X_test, test_labels, train_time_s=train_time
        )

        # ── Save model ────────────────────────────────────────────────────────
        trainer.save()

        # ── Save evaluation artifacts ─────────────────────────────────────────
        evaluator.save_artifacts(
            np.asarray(test_labels), y_pred, y_prob, metrics
        )

        all_metrics.append(metrics)

    return all_metrics


def print_comparison_table(all_metrics: list[dict]) -> None:
    """Print a ranked comparison table sorted by F1 macro."""
    df = pd.DataFrame(all_metrics)
    cols = ["model", "accuracy", "f1_macro", "roc_auc",
            "train_time_s", "infer_ms_per_sample"]
    df = df[cols].sort_values("f1_macro", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(" CLASSICAL ML RESULTS — amazon_polarity")
    print("=" * 70)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 70 + "\n")


def save_comparison(all_metrics: list[dict]) -> pathlib.Path:
    out = RESULTS_DIR / "comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Comparison saved → %s", out)
    return out


def main() -> None:
    logger.info("Classical ML training pipeline starting …")
    all_metrics = run_classical_pipeline()
    print_comparison_table(all_metrics)
    save_comparison(all_metrics)
    logger.info("Classical ML pipeline complete.")


if __name__ == "__main__":
    main()
