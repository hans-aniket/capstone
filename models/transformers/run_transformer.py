"""
models/transformers/run_transformer.py
───────────────────────────────────────
End-to-end orchestration for fine-tuning a Transformer model.
"""

import sys
import argparse
import random
import logging
import pathlib
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import SPLITS_DIR, SEED, TRANSFORMER_MODELS
from preprocessing.transformer_preprocessor import TransformerPreprocessor

from models.transformers.config import (
    DEVICE, TRAIN_CONFIG, get_model_paths, NUM_CLASSES
)
from models.transformers.dataset import build_dataloaders, dataset_info
from models.transformers.trainer import TransformerTrainer
from models.transformers.evaluator import TransformerEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_transformer")


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
            raise FileNotFoundError(f"Split not found: {path}")
    return (
        pd.read_parquet(SPLITS_DIR / "train.parquet"),
        pd.read_parquet(SPLITS_DIR / "test.parquet"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["bert", "distilbert"], required=True,
                        help="Which transformer model to train")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Optionally train on fewer samples for quick testing")
    args = parser.parse_args()

    model_key = args.model
    hf_model_name = TRANSFORMER_MODELS[model_key]
    paths = get_model_paths(model_key)
    
    display_name = "BERT" if model_key == "bert" else "DistilBERT"

    logger.info("=" * 58)
    logger.info("%s Training Pipeline | device: %s", display_name, DEVICE)
    logger.info("=" * 58)

    lock_seeds()

    # ── 1. Load splits ────────────────────────────────────────────────────────
    train_df, test_df = load_splits()
    
    if args.subsample > 0:
        logger.info("Subsampling data to %d rows for testing...", args.subsample)
        train_df = train_df.sample(n=args.subsample, random_state=SEED)
        test_df = test_df.sample(n=min(args.subsample, len(test_df)), random_state=SEED)

    logger.info("Splits | train: %d  test: %d", len(train_df), len(test_df))
    
    train_texts, train_labels = train_df["text"].tolist(), train_df["label"].tolist()
    test_texts, test_labels   = test_df["text"].tolist(), test_df["label"].tolist()

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 1 | Tokenization")

    tp = TransformerPreprocessor(hf_model_name)
    train_ds = tp.prepare(train_texts, train_labels)
    test_ds  = tp.prepare(test_texts, test_labels)
    
    tp.save_tokenizer()
    tp.save_config()

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 2 | DataLoaders")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, test_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        val_split=TRAIN_CONFIG["val_split"],
        seed=SEED,
    )
    logger.info("%s", dataset_info(train_loader, val_loader, test_loader))

    # ── 4. Model ──────────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 3 | Initializing %s", display_name)

    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=NUM_CLASSES)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %d  device: %s", params, DEVICE)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 4 | Training")

    trainer = TransformerTrainer(
        model        = model,
        model_name   = display_name,
        ckpt_best    = paths["ckpt_best"],
        ckpt_last    = paths["ckpt_last"],
        history_path = paths["history"],
        results_dir  = paths["results_dir"],
        device       = DEVICE,
        lr           = TRAIN_CONFIG["lr"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        clip_norm    = TRAIN_CONFIG["clip_grad_norm"],
        patience     = TRAIN_CONFIG["patience"],
    )
    
    history = trainer.train(train_loader, val_loader, TRAIN_CONFIG["num_epochs"])
    train_time_s = history["total_train_time_s"]

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info("-" * 58)
    logger.info("STAGE 5 | Test Evaluation")

    trainer.load_best_checkpoint()
    evaluator = TransformerEvaluator(trainer, display_name, paths["results_dir"])
    metrics, y_pred, y_prob = evaluator.evaluate(test_loader, test_labels, train_time_s)
    evaluator.save_artifacts(np.asarray(test_labels), y_pred, y_prob, metrics)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 58)
    logger.info("%s RESULTS", display_name)
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  F1 Macro  : %.4f", metrics["f1_macro"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("  Train time: %.1fs", metrics["train_time_s"])
    logger.info("  Infer     : %.4f ms/sample", metrics["infer_ms_per_sample"])
    logger.info("=" * 58)

if __name__ == "__main__":
    main()
