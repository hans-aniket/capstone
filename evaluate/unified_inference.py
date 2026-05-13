"""
evaluate/unified_inference.py
─────────────────────────────
Unified comparative inference pipeline.
Loads all 6 specified models (+ optionally Transformers) into memory simultaneously
and passes a raw string through all of them, comparing their latencies, predictions,
and confidence scores side-by-side.
"""

import sys
import logging
import pathlib
import time
import os

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.classical.predictor import ClassicalPredictor
from models.deep_learning.dl_predictor import DLPredictor
from models.deep_learning.lstm_model import LSTMClassifier
from models.deep_learning.cnn_model import TextCNN
from models.deep_learning.cnn_lstm_model import CNNLSTMClassifier
from models.transformers.predictor import TransformerPredictor

# Config paths
from models.deep_learning.config import (
    LSTM_CHECKPOINT_BEST, LSTM_MODEL_CONFIG_FILE,
    CNN_CHECKPOINT_BEST, CNN_MODEL_CONFIG_FILE,
    CNN_LSTM_CHECKPOINT_BEST, CNN_LSTM_MODEL_CONFIG_FILE
)
from models.transformers.config import get_model_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("unified_inference")


class UnifiedPredictor:
    """
    Loads all trained models into memory to perform simultaneous comparative inference.
    """
    def __init__(self):
        self.predictors = {}
        self._load_all_models()

    def _load_all_models(self):
        logger.info("Initializing Unified Comparative Inference System...")
        
        # 1-3: Classical ML
        logger.info("Loading Classical Models...")
        for name, key in [("Naive Bayes", "naive_bayes"), ("Logistic Regression", "logistic_regression"), ("SVM", "svm")]:
            try:
                self.predictors[name] = ClassicalPredictor.from_saved(key)
            except Exception as e:
                logger.warning("Could not load %s: %s", name, e)

        # To prevent OOM (Out of Memory) crashes on Render's 512MB Free Tier,
        # we skip loading the heavy PyTorch Deep Learning and Transformer models.
        if os.environ.get("RENDER") == "true":
            logger.warning("Detected Render environment! Skipping Deep Learning and Transformer models to prevent OOM crash.")
            return

        # 4: LSTM
        logger.info("Loading Deep Learning (LSTM)...")
        try:
            self.predictors["LSTM"] = DLPredictor.from_saved(
                model_cls=LSTMClassifier,
                checkpoint_path=LSTM_CHECKPOINT_BEST,
                model_config_path=LSTM_MODEL_CONFIG_FILE,
                model_name="LSTM"
            )
        except Exception as e:
            logger.warning("Could not load LSTM: %s", e)

        # 5: TextCNN
        logger.info("Loading Deep Learning (TextCNN)...")
        try:
            self.predictors["TextCNN"] = DLPredictor.from_saved(
                model_cls=TextCNN,
                checkpoint_path=CNN_CHECKPOINT_BEST,
                model_config_path=CNN_MODEL_CONFIG_FILE,
                model_name="TextCNN"
            )
        except Exception as e:
            logger.warning("Could not load TextCNN: %s", e)

        # 6: CNN-LSTM
        logger.info("Loading Deep Learning (CNN-LSTM)...")
        try:
            self.predictors["CNN-LSTM"] = DLPredictor.from_saved(
                model_cls=CNNLSTMClassifier,
                checkpoint_path=CNN_LSTM_CHECKPOINT_BEST,
                model_config_path=CNN_LSTM_MODEL_CONFIG_FILE,
                model_name="CNN-LSTM"
            )
        except Exception as e:
            logger.warning("Could not load CNN-LSTM: %s", e)

        # Optional: DistilBERT
        logger.info("Loading Transformers (DistilBERT)...")
        try:
            distilbert_paths = get_model_paths("distilbert")
            self.predictors["DistilBERT"] = TransformerPredictor.from_saved(
                hf_model_name="distilbert-base-uncased",
                checkpoint_path=distilbert_paths["ckpt_best"],
                model_name="DistilBERT"
            )
        except Exception as e:
            logger.warning("Could not load DistilBERT: %s", e)

        logger.info("System Ready. Loaded %d models.", len(self.predictors))

    def predict_single(self, text: str) -> dict:
        logger.info("\n" + "="*80)
        logger.info("INPUT TEXT: \"%s\"", text)
        logger.info("="*80)
        
        results = {}
        for name, predictor in self.predictors.items():
            try:
                res = predictor.predict_single(text)
                results[name] = res
                logger.info(
                    "%-20s | Label: %-8s | Conf: %5.2f%% | Latency: %6.2f ms",
                    name, res["label_name"].upper(), res["confidence"] * 100, res["latency_ms"]
                )
            except Exception as e:
                logger.error("Error running %s: %s", name, e)
                
        logger.info("="*80 + "\n")
        return results

    def start_repl(self):
        print("\n\n" + "*" * 60)
        print("  UNIFIED COMPARATIVE NLP SYSTEM - INTERACTIVE INFERENCE  ")
        print("*" * 60)
        print("Type your product review below and hit Enter.")
        print("Type 'exit' or 'quit' to close.\n")
        
        while True:
            try:
                text = input("\nReview > ")
                if text.strip().lower() in ['exit', 'quit']:
                    break
                if not text.strip():
                    continue
                self.predict_single(text)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    system = UnifiedPredictor()
    system.predict_single("I absolutely love this product! It exceeds all my expectations and the build quality is fantastic.")
    system.predict_single("This broke after just two days. The customer service was terrible and refused to give me a refund. Waste of money.")
    # Uncomment to run interactive REPL:
    # system.start_repl()
