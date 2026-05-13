"""
evaluate_cnn_lstm.py
──────────────────────
Manually evaluate the saved CNN-LSTM checkpoint to complete the pipeline.
"""

import sys
import logging
import pathlib
import pandas as pd
import numpy as np
import torch

_PROJECT_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.deep_learning.cnn_lstm_model import CNNLSTMClassifier, load_cnn_lstm_config
from models.deep_learning.dl_trainer import DLTrainer
from models.deep_learning.dl_evaluator import DLEvaluator
from models.deep_learning.dl_predictor import DLPredictor
from models.deep_learning.dataset import build_dataloaders
from preprocessing.dl_preprocessor import Vocabulary, DLPreprocessor
from preprocessing.config import SPLITS_DIR
from models.deep_learning.config import (
    MAX_LEN, DEVICE, TRAIN_CONFIG, 
    CNN_LSTM_CHECKPOINT_BEST, CNN_LSTM_CHECKPOINT_LAST, 
    CNN_LSTM_TRAINING_HISTORY, CNN_LSTM_RESULTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate_cnn_lstm")

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
    cfg = load_cnn_lstm_config()
    model = CNNLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_filters=cfg["num_filters"],
        kernel_size=cfg["kernel_size"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        pad_idx=cfg["pad_idx"]
    )

    logger.info("Initializing Trainer and Loading Best Checkpoint...")
    trainer = DLTrainer(
        model=model,
        model_name="CNN-LSTM",
        ckpt_best=CNN_LSTM_CHECKPOINT_BEST,
        ckpt_last=CNN_LSTM_CHECKPOINT_LAST,
        history_path=CNN_LSTM_TRAINING_HISTORY,
        results_dir=CNN_LSTM_RESULTS_DIR,
        device=DEVICE
    )
    trainer.load_best_checkpoint()

    logger.info("Running Evaluation Suite...")
    evaluator = DLEvaluator(trainer, "CNN-LSTM", CNN_LSTM_RESULTS_DIR)
    metrics, y_pred, y_prob = evaluator.evaluate(test_loader, test_labels, train_time_s=895.0) # Sum of times for 3 epochs
    evaluator.save_artifacts(np.asarray(test_labels), y_pred, y_prob, metrics)

    logger.info("Testing DLPredictor interface...")
    predictor = DLPredictor.from_saved(
        model_cls=CNNLSTMClassifier,
        checkpoint_path=CNN_LSTM_CHECKPOINT_BEST,
        model_config_path=cfg["_dummy_path"] if False else pathlib.Path("models/deep_learning/saved/cnn_lstm/model_config.json"),
        model_name="CNN-LSTM"
    )
    sample_text = "This product is an absolute waste of money. Do not buy!"
    result = predictor.predict_single(sample_text)
    
    logger.info("Inference Result: %s", result)
    logger.info("CNN-LSTM Verification Complete!")

if __name__ == "__main__":
    main()
