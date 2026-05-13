"""
models/deep_learning/predictor.py
───────────────────────────────────
LSTMPredictor: unified BasePredictor interface for the LSTM model.

This is the object the comparative system calls at inference time.
It loads the saved model + vocabulary, applies DL preprocessing to
raw text strings, and returns predictions in the standard schema
shared by ClassicalPredictor, CNN-LSTMPredictor, and TransformerPredictor.
"""

import time
import json
import logging
import pathlib

import numpy as np
import torch

from models.base import BasePredictor
from models.deep_learning.lstm_model import LSTMClassifier, load_model_config
from models.deep_learning.config import (
    DEVICE, MAX_LEN,
    LSTM_CHECKPOINT_BEST, LSTM_SAVED_DIR,
)
from preprocessing.dl_preprocessor import Vocabulary
from preprocessing.config import LABEL_MAP

logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """
    Loads a saved LSTM checkpoint and vocabulary for raw-text inference.

    All public methods return the same dict schema as ClassicalPredictor,
    enabling the comparative harness to call them interchangeably.

    Usage:
        predictor = LSTMPredictor.from_saved()
        result    = predictor.predict_single("This product is amazing!")
        results   = predictor.predict(["Great!", "Terrible quality."])
    """

    model_name: str = "lstm"
    tier:       str = "deep_learning"

    def __init__(
        self,
        model:      LSTMClassifier,
        vocabulary: Vocabulary,
        max_len:    int           = MAX_LEN,
        device:     torch.device  = DEVICE,
    ):
        self.model      = model.to(device)
        self.model.eval()
        self.vocabulary = vocabulary
        self.max_len    = max_len
        self.device     = device

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        checkpoint_path: pathlib.Path | None = None,
        device:          torch.device        = DEVICE,
    ) -> "LSTMPredictor":
        """
        Load LSTM model from the best checkpoint and vocabulary from disk.

        Args:
            checkpoint_path: Override path (defaults to LSTM_CHECKPOINT_BEST).
            device:          Target device (CPU/GPU).
        """
        ckpt_path = checkpoint_path or LSTM_CHECKPOINT_BEST
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No LSTM checkpoint at {ckpt_path}.\n"
                "Run:  python models/deep_learning/run_lstm.py  first."
            )

        # Load model config saved during training
        cfg   = load_model_config()
        model = LSTMClassifier(
            vocab_size    = cfg["vocab_size"],
            embed_dim     = cfg["embed_dim"],
            hidden_dim    = cfg["hidden_dim"],
            num_layers    = cfg["num_layers"],
            num_classes   = cfg["num_classes"],
            dropout       = cfg["dropout"],
            bidirectional = cfg["bidirectional"],
            pad_idx       = cfg["pad_idx"],
        )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "[LSTMPredictor] Loaded checkpoint (epoch=%d  val_f1=%.4f)",
            ckpt.get("epoch", -1), ckpt.get("val_f1", 0.0),
        )

        vocab = Vocabulary.load()
        max_len = cfg.get("max_len", MAX_LEN)
        return cls(model, vocab, max_len, device)

    # ── BasePredictor interface ────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> dict:
        """
        Run batch inference on raw review strings.

        Returns:
            {
              "labels":        list[int],
              "label_names":   list[str],
              "confidences":   list[float],   P(positive)
              "latency_ms":    float,
              "ms_per_sample": float,
            }
        """
        t0       = time.perf_counter()
        seqs     = self.vocabulary.encode_batch(texts, self.max_len)   # (N, max_len)
        x        = torch.tensor(seqs, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()

        labels  = proba.argmax(axis=1).tolist()
        confs   = proba[:, 1].tolist()
        elapsed = (time.perf_counter() - t0) * 1000   # ms

        return {
            "labels":        labels,
            "label_names":   [LABEL_MAP[l] for l in labels],
            "confidences":   [round(c, 4) for c in confs],
            "latency_ms":    round(elapsed, 3),
            "ms_per_sample": round(elapsed / max(len(texts), 1), 4),
        }

    def predict_single(self, text: str) -> dict:
        """
        Run inference on a single raw review string.

        Returns:
            {
              "label":      int,
              "label_name": str,
              "confidence": float,
              "latency_ms": float,
            }
        """
        result = self.predict([text])
        return {
            "label":      result["labels"][0],
            "label_name": result["label_names"][0],
            "confidence": result["confidences"][0],
            "latency_ms": result["latency_ms"],
        }

    def __repr__(self) -> str:
        return f"LSTMPredictor(device='{self.device}', vocab_size={len(self.vocabulary)})"
