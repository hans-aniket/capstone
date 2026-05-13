"""
models/deep_learning/run_cnn.py
────────────────────────────────
TextCNN training pipeline orchestration.

Run from project root:
    python models/deep_learning/run_cnn.py

Outputs:
    models/deep_learning/saved/cnn/checkpoint_best.pt
    models/deep_learning/saved/cnn/checkpoint_last.pt
    models/deep_learning/saved/cnn/training_history.json
    models/deep_learning/saved/cnn/model_config.json
    results/deep_learning/cnn/metrics.json
    results/deep_learning/cnn/classification_report.txt
    results/deep_learning/cnn/confusion_matrix.png
    results/deep_learning/cnn/training_curves.png
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
    DEVICE, TRAIN_CONFIG, CNN_CONFIG, EMBED_DIM, MAX_LEN,
    CNN_CHECKPOINT_BEST, CNN_CHECKPOINT_LAST,
    CNN_TRAINING_HISTORY, CNN_RESULTS_DIR,
)
from models.deep_learning.dataset import build_dataloaders, dataset_info
from models.deep_learning.lstm_model import load_glove          # GloVe loader is model-agnostic
from models.deep_learning.cnn_model import TextCNN, save_cnn_config
from models.deep_learning.dl_trainer import DLTrainer
from models.deep_learning.dl_evaluator import DLEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_cnn")

MODEL_NAME = "TextCNN"


def lock_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [SPLITS_DIR / "train.parquet", SPLITS_DIR / "test.parquet"]:
        if not path.exists():
            raise FileNotFoundError(
                f"Split not found: {path}\nRun:  python data/prepare_data.py"
            )
    return (
        pd.read_parquet(SPLITS_DIR / "train.parquet"),
        pd.read_parquet(SPLITS_DIR / "test.parquet"),
    )


def main() -> None:
    logger.info("=" * 58)
    logger.info("%s Training Pipeline | device: %s", MODEL_NAME, DEVICE)
    logger.info("=" * 58)

    lock_seeds()

    # ── 1. Load splits ────────────────────────────────────────────────────────
    train_df, test_df = load_splits()
    logger.info("Splits | train: %d  test: %d", len(train_df), len(test_df))

    train_texts  = train_df["text"].tolist()
    test_texts   = test_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # ── 2. DL preprocessing ───────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 1 | DL Preprocessing")

    dp = DLPreprocessor(min_freq=2, max_len=MAX_LEN)
    X_train = dp.fit_transform(train_texts)
    X_test  = dp.transform(test_texts)
    dp.save()
    logger.info("Vocab: %d  train: %s  test: %s",
                dp.vocab_size, X_train.shape, X_test.shape)

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 2 | DataLoaders")

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, train_labels, X_test, test_labels,
        batch_size=TRAIN_CONFIG["batch_size"],
        val_split=TRAIN_CONFIG["val_split"],
        seed=SEED,
    )
    logger.info("%s", dataset_info(train_loader, val_loader, test_loader))

    # ── 4. Model ──────────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 3 | Initializing %s", MODEL_NAME)

    pretrained = load_glove(dp.vocabulary.token2idx, embed_dim=EMBED_DIM)
    logger.info("GloVe: %s", "loaded" if pretrained is not None else "not found, using random")

    model = TextCNN(
        vocab_size  = dp.vocab_size,
        embed_dim   = EMBED_DIM,
        pretrained_embeddings = pretrained,
        **CNN_CONFIG,
    )
    save_cnn_config(dp.vocab_size)
    logger.info("Params: %d  device: %s", model.count_parameters(), DEVICE)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 4 | Training")

    trainer = DLTrainer(
        model        = model,
        model_name   = MODEL_NAME,
        ckpt_best    = CNN_CHECKPOINT_BEST,
        ckpt_last    = CNN_CHECKPOINT_LAST,
        history_path = CNN_TRAINING_HISTORY,
        results_dir  = CNN_RESULTS_DIR,
        device       = DEVICE,
        lr           = TRAIN_CONFIG["lr"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        clip_norm    = TRAIN_CONFIG["clip_grad_norm"],
        patience     = TRAIN_CONFIG["patience"],
    )
    history      = trainer.train(train_loader, val_loader, TRAIN_CONFIG["num_epochs"])
    train_time_s = history["total_train_time_s"]

    # ── 6. Load best checkpoint + test evaluation ──────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 5 | Test Evaluation")

    trainer.load_best_checkpoint()
    evaluator = DLEvaluator(trainer, MODEL_NAME, CNN_RESULTS_DIR)
    metrics, y_pred, y_prob = evaluator.evaluate(test_loader, test_labels, train_time_s)
    evaluator.save_artifacts(np.asarray(test_labels), y_pred, y_prob, metrics)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 58)
    logger.info("%s RESULTS", MODEL_NAME)
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  F1 Macro  : %.4f", metrics["f1_macro"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("  Train time: %.1fs", metrics["train_time_s"])
    logger.info("  Infer     : %.4f ms/sample", metrics["infer_ms_per_sample"])
    logger.info("=" * 58)

    # Cross-model comparison (if prior results exist)
    classical_path = _PROJECT_ROOT / "results" / "classical" / "comparison.json"
    if classical_path.exists():
        with open(classical_path) as f:
            rows = json.load(f)
        rows.append(metrics)
        df = pd.DataFrame(rows)[
            ["model", "accuracy", "f1_macro", "roc_auc",
             "train_time_s", "infer_ms_per_sample"]
        ].sort_values("f1_macro", ascending=False)
        print("\n" + "=" * 70)
        print("  COMPARISON (classical + CNN)")
        print("=" * 70)
        print(df.to_string(index=False, float_format="{:.4f}".format))
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
