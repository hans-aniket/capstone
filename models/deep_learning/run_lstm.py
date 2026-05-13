"""
models/deep_learning/run_lstm.py
─────────────────────────────────
Full LSTM training pipeline orchestration.

Run from project root:
    python models/deep_learning/run_lstm.py

Pipeline:
    1. Load train/test Parquet splits
    2. DL preprocessing: build vocabulary (train only), encode sequences
    3. Build train / val / test DataLoaders
    4. Initialize LSTMClassifier (+ optional GloVe embeddings)
    5. Train with early stopping — checkpoint best model by val F1
    6. Load best checkpoint
    7. Evaluate on test set with full metric suite
    8. Save all artifacts
    9. Print comparison vs classical ML results

Outputs:
    models/deep_learning/saved/lstm/checkpoint_best.pt
    models/deep_learning/saved/lstm/checkpoint_last.pt
    models/deep_learning/saved/lstm/training_history.json
    models/deep_learning/saved/lstm/model_config.json
    results/deep_learning/lstm/metrics.json
    results/deep_learning/lstm/classification_report.txt
    results/deep_learning/lstm/confusion_matrix.png
    results/deep_learning/lstm/training_curves.png
"""

import sys
import json
import random
import logging
import pathlib

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import SPLITS_DIR, SEED
from preprocessing.dl_preprocessor import DLPreprocessor

from models.deep_learning.config import (
    DEVICE, TRAIN_CONFIG, LSTM_CONFIG,
    EMBED_DIM, MAX_LEN, LSTM_RESULTS_DIR,
)
from models.deep_learning.dataset import build_dataloaders, dataset_info
from models.deep_learning.lstm_model import LSTMClassifier, load_glove, save_model_config
from models.deep_learning.trainer import LSTMTrainer
from models.deep_learning.evaluator import LSTMEvaluator
from models.shared.metrics import save_metrics_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_lstm")


# ── Seed locking ──────────────────────────────────────────────────────────────
def lock_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 58)
    logger.info("LSTM Training Pipeline | device: %s", DEVICE)
    logger.info("=" * 58)

    lock_seeds()

    # ── 1. Load splits ────────────────────────────────────────────────────────
    train_df, test_df = load_splits()
    train_texts  = train_df["text"].tolist()
    test_texts   = test_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # ── 2. DL preprocessing ───────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 1 | DL Preprocessing")

    dp = DLPreprocessor(min_freq=2, max_len=MAX_LEN)

    # Report token length stats to validate MAX_LEN choice
    stats = dp.length_stats(train_texts)
    logger.info("Token length stats: %s", stats)

    X_train = dp.fit_transform(train_texts)   # (20000, 200) int32
    X_test  = dp.transform(test_texts)        # ( 5000, 200) int32
    dp.save()                                  # vocabulary.pkl + vocab_config.json

    logger.info("Sequences | train: %s  test: %s  vocab_size: %d",
                X_train.shape, X_test.shape, dp.vocab_size)

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 2 | Building DataLoaders")

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, train_labels, X_test, test_labels,
        batch_size=TRAIN_CONFIG["batch_size"],
        val_split=TRAIN_CONFIG["val_split"],
        seed=SEED,
    )
    info = dataset_info(train_loader, val_loader, test_loader)
    logger.info("DataLoaders | %s", info)

    # ── 4. Model initialization ───────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 3 | Initializing LSTMClassifier")

    pretrained = load_glove(dp.vocabulary.token2idx, embed_dim=EMBED_DIM)
    if pretrained is not None:
        logger.info("GloVe embeddings loaded.")
    else:
        logger.info("GloVe not found — using random embeddings.")

    model = LSTMClassifier(
        vocab_size    = dp.vocab_size,
        embed_dim     = EMBED_DIM,
        pretrained_embeddings = pretrained,
        **LSTM_CONFIG,
    )
    logger.info("Model | params: %d  device: %s", model.count_parameters(), DEVICE)

    # Save model config for later loading by LSTMPredictor
    save_model_config(dp.vocab_size)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 4 | Training")

    trainer = LSTMTrainer(
        model,
        device       = DEVICE,
        lr           = TRAIN_CONFIG["lr"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        clip_norm    = TRAIN_CONFIG["clip_grad_norm"],
        patience     = TRAIN_CONFIG["patience"],
    )
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=TRAIN_CONFIG["num_epochs"],
    )
    total_train_time = history["total_train_time_s"]

    # ── 6. Load best checkpoint ───────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 5 | Loading best checkpoint")
    trainer.load_best_checkpoint()

    # ── 7. Test evaluation ────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 6 | Test Evaluation")

    evaluator = LSTMEvaluator(trainer)
    metrics, y_pred, y_prob = evaluator.evaluate(
        test_loader, test_labels, train_time_s=total_train_time
    )
    evaluator.save_artifacts(np.asarray(test_labels), y_pred, y_prob, metrics)

    # ── 8. Final report ───────────────────────────────────────────────────────
    logger.info("=" * 58)
    logger.info("LSTM RESULTS")
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  F1 Macro  : %.4f", metrics["f1_macro"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("  Train time: %.1fs", metrics["train_time_s"])
    logger.info("  Infer     : %.4f ms/sample", metrics["infer_ms_per_sample"])
    logger.info("=" * 58)

    # ── 9. Compare vs classical ML (if available) ─────────────────────────────
    classical_path = _PROJECT_ROOT / "results" / "classical" / "comparison.json"
    if classical_path.exists():
        with open(classical_path) as f:
            classical = json.load(f)
        print("\n" + "=" * 68)
        print("  FULL COMPARISON — amazon_polarity (5 000 test samples)")
        print("=" * 68)
        rows = classical + [metrics]
        df   = pd.DataFrame(rows)[
            ["model", "accuracy", "f1_macro", "roc_auc",
             "train_time_s", "infer_ms_per_sample"]
        ].sort_values("f1_macro", ascending=False)
        print(df.to_string(index=False, float_format="{:.4f}".format))
        print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
