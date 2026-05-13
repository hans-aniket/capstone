"""
Script to manually evaluate the saved LSTM checkpoint and test the predictor.
This verifies the LSTM pipeline end-to-end without needing to wait for a full training run.
"""

import sys
import logging
import pathlib
import pandas as pd
import numpy as np
import torch

_PROJECT_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.deep_learning.lstm_model import LSTMClassifier, load_model_config
from models.deep_learning.trainer import LSTMTrainer
from models.deep_learning.evaluator import LSTMEvaluator
from models.deep_learning.predictor import LSTMPredictor
from models.deep_learning.dataset import build_dataloaders
from preprocessing.dl_preprocessor import Vocabulary, DLPreprocessor
from preprocessing.config import SPLITS_DIR
from models.deep_learning.config import MAX_LEN, DEVICE, TRAIN_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate_lstm")

def main():
    logger.info("Loading test data...")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    logger.info("Loading DL Preprocessor...")
    dp = DLPreprocessor(min_freq=2, max_len=MAX_LEN)
    dp.vocabulary = Vocabulary.load()
    X_test = dp.transform(test_texts)

    logger.info("Building DataLoader...")
    _, _, test_loader = build_dataloaders(
        X_test[:2], test_labels[:2], X_test, test_labels,
        batch_size=TRAIN_CONFIG["batch_size"]
    )

    logger.info("Loading Model Config...")
    cfg = load_model_config()
    model = LSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        pad_idx=cfg["pad_idx"]
    )

    logger.info("Initializing Trainer and Loading Best Checkpoint...")
    trainer = LSTMTrainer(model, device=DEVICE)
    trainer.load_best_checkpoint()

    logger.info("Running Evaluation Suite...")
    evaluator = LSTMEvaluator(trainer)
    # Passed train_time_s=1329 as that was what epoch 1 took
    metrics, y_pred, y_prob = evaluator.evaluate(test_loader, test_labels, train_time_s=1329.0)
    evaluator.save_artifacts(np.asarray(test_labels), y_pred, y_prob, metrics)

    logger.info("Testing LSTMPredictor interface...")
    predictor = LSTMPredictor.from_saved()
    sample_text = "This product was absolutely terrible. I want a refund."
    result = predictor.predict_single(sample_text)
    
    logger.info("Inference Result: %s", result)
    logger.info("LSTM Verification Complete!")

if __name__ == "__main__":
    main()
