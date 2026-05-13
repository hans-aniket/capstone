"""
preprocessing/run_preprocessing.py
────────────────────────────────────
Master orchestration script.
Runs the full preprocessing pipeline for all three model tiers
and saves every artifact needed for downstream training scripts.

Run from project root:
    python preprocessing/run_preprocessing.py

Outputs (in preprocessing/artifacts/):
    tfidf_vectorizer.joblib          ← Classical ML
    vocabulary.pkl                   ← LSTM / CNN-LSTM
    vocab_config.json                ← LSTM / CNN-LSTM (human-readable)
    tokenizers/bert_base_uncased/    ← BERT tokenizer files
    tokenizers/distilbert_base_uncased/ ← DistilBERT tokenizer files
    preprocessing_log.json           ← Run metadata for reproducibility
"""

import sys
import json
import logging
import pathlib
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path when run directly ──────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import (
    SEED, SPLITS_DIR, ARTIFACTS_DIR,
    TFIDF_CONFIG, DL_MAX_LEN, DL_MIN_FREQ, TRANSFORMER_MAX_LEN,
    TRANSFORMER_MODELS, ARTIFACT_SPLITS_META,
)
from preprocessing.global_cleaner import clean_dataframe
from preprocessing.classical_preprocessor import ClassicalPreprocessor
from preprocessing.dl_preprocessor import DLPreprocessor
from preprocessing.transformer_preprocessor import TransformerPreprocessor
from preprocessing.artifact_manager import save_json, save_preprocessing_log, list_artifacts

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocessing")


# ── Seed locking ──────────────────────────────────────────────────────────────
def lock_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeds locked: random=%d  numpy=%d", seed, seed)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = SPLITS_DIR / "train.parquet"
    test_path  = SPLITS_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {SPLITS_DIR}.\n"
            "Run:  python data/prepare_data.py  first."
        )

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    logger.info("Loaded splits | train: %d  test: %d", len(train_df), len(test_df))
    return train_df, test_df


# ── Stage 0: Global cleaning ──────────────────────────────────────────────────
def stage_global_cleaning(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("─" * 55)
    logger.info("STAGE 0 | Global cleaning")
    train_df = clean_dataframe(train_df, text_col="text", drop_invalid=True)
    test_df  = clean_dataframe(test_df,  text_col="text", drop_invalid=True)
    logger.info("After cleaning | train: %d  test: %d", len(train_df), len(test_df))
    return train_df, test_df


# ── Stage 1: Classical ML ─────────────────────────────────────────────────────
def stage_classical(
    train_texts: list[str],
    test_texts:  list[str],
) -> dict:
    logger.info("─" * 55)
    logger.info("STAGE 1 | Classical ML  (TF-IDF)")

    cp = ClassicalPreprocessor(config=TFIDF_CONFIG)
    X_train = cp.fit_transform(train_texts)
    X_test  = cp.transform(test_texts)
    cp.save()

    logger.info(
        "TF-IDF | train: %s  test: %s  vocab: %d",
        X_train.shape, X_test.shape, cp.vocab_size,
    )
    return {
        "vocab_size":   cp.vocab_size,
        "train_shape":  list(X_train.shape),
        "test_shape":   list(X_test.shape),
        "ngram_range":  list(TFIDF_CONFIG["ngram_range"]),
        "max_features": TFIDF_CONFIG["max_features"],
    }


# ── Stage 2: Deep Learning ────────────────────────────────────────────────────
def stage_dl(
    train_texts: list[str],
    test_texts:  list[str],
) -> dict:
    logger.info("─" * 55)
    logger.info("STAGE 2 | Deep Learning  (LSTM / CNN-LSTM)")

    dp = DLPreprocessor(min_freq=DL_MIN_FREQ, max_len=DL_MAX_LEN)

    # Report length statistics so DL_MAX_LEN can be validated
    stats  = dp.length_stats(train_texts)
    logger.info("Token length percentiles: %s", stats)

    X_train = dp.fit_transform(train_texts)
    X_test  = dp.transform(test_texts)
    dp.save()

    logger.info(
        "DL sequences | train: %s  test: %s  vocab: %d",
        X_train.shape, X_test.shape, dp.vocab_size,
    )
    return {
        "vocab_size":    dp.vocab_size,
        "train_shape":   list(X_train.shape),
        "test_shape":    list(X_test.shape),
        "max_len":       DL_MAX_LEN,
        "min_freq":      DL_MIN_FREQ,
        "length_stats":  stats,
    }


# ── Stage 3: Transformers ─────────────────────────────────────────────────────
def stage_transformers(
    train_texts:  list[str],
    test_texts:   list[str],
    train_labels: list[int],
    test_labels:  list[int],
) -> dict:
    logger.info("─" * 55)
    logger.info("STAGE 3 | Transformers  (BERT / DistilBERT)")

    results = {}
    for key, model_name in TRANSFORMER_MODELS.items():
        logger.info("  Processing: %s …", model_name)
        tp = TransformerPreprocessor(model_name=model_name, max_length=TRANSFORMER_MAX_LEN)

        train_ds = tp.prepare(train_texts, train_labels)
        test_ds  = tp.prepare(test_texts,  test_labels)

        tp.save_tokenizer()
        tp.save_config()

        results[key] = {
            "model_name":      model_name,
            "max_length":      TRANSFORMER_MAX_LEN,
            "vocab_size":      tp.tokenizer.vocab_size,
            "train_n_samples": len(train_ds),
            "test_n_samples":  len(test_ds),
        }
        logger.info("  %s done | train: %d  test: %d", key, len(train_ds), len(test_ds))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 55)
    logger.info("Preprocessing pipeline starting …")
    logger.info("Artifacts directory: %s", ARTIFACTS_DIR)

    lock_seeds()

    # Load splits
    train_df, test_df = load_splits()
    train_texts  = train_df["text"].tolist()
    test_texts   = test_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # Stage 0 — global cleaning
    train_df, test_df = stage_global_cleaning(train_df, test_df)
    # Refresh lists after cleaning (may have dropped rows)
    train_texts  = train_df["text"].tolist()
    test_texts   = test_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # Stage 1 — classical ML
    classical_meta = stage_classical(train_texts, test_texts)

    # Stage 2 — deep learning
    dl_meta = stage_dl(train_texts, test_texts)

    # Stage 3 — transformers
    transformer_meta = stage_transformers(train_texts, test_texts, train_labels, test_labels)

    # Save full preprocessing log
    log_entry = {
        "seed":          SEED,
        "train_n":       len(train_texts),
        "test_n":        len(test_texts),
        "classical":     classical_meta,
        "deep_learning": dl_meta,
        "transformers":  transformer_meta,
    }
    save_preprocessing_log(log_entry)

    # Print artifact inventory
    logger.info("=" * 55)
    logger.info("Preprocessing complete. Artifact inventory:")
    list_artifacts()
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
