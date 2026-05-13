"""
models/deep_learning/dl_predictor.py
──────────────────────────────────────
Generic DLPredictor — implements BasePredictor for any vocabulary-based
deep learning text classifier.

Both CNNPredictor and CNNLSTMPredictor are thin wrappers around this class,
providing the same dict schema as ClassicalPredictor and LSTMPredictor.

Usage:
    # Build a predictor for a saved CNN model:
    predictor = DLPredictor.from_saved(
        model_cls        = TextCNN,
        checkpoint_path  = CNN_CHECKPOINT_BEST,
        model_config_path= CNN_MODEL_CONFIG_FILE,
        model_name       = "CNN",
    )
    result = predictor.predict_single("Great product!")
"""

import json
import time
import logging
import pathlib
from typing import Type

import torch
import numpy as np

from models.base import BasePredictor
from models.deep_learning.config import DEVICE, MAX_LEN
from preprocessing.dl_preprocessor import Vocabulary
from preprocessing.config import LABEL_MAP

logger = logging.getLogger(__name__)


class DLPredictor(BasePredictor):
    """
    Generic raw-text inference interface for vocabulary-based DL models.
    Implements BasePredictor — identical public API to ClassicalPredictor
    and LSTMPredictor for seamless integration in the comparative system.
    """

    tier: str = "deep_learning"

    def __init__(
        self,
        model:      torch.nn.Module,
        vocabulary: Vocabulary,
        model_name: str,
        max_len:    int          = MAX_LEN,
        device:     torch.device = DEVICE,
    ):
        self.model      = model.to(device)
        self.model.eval()
        self.vocabulary = vocabulary
        self.model_name = model_name
        self.max_len    = max_len
        self.device     = device

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        model_cls:         Type[torch.nn.Module],
        checkpoint_path:   pathlib.Path,
        model_config_path: pathlib.Path,
        model_name:        str,
        device:            torch.device = DEVICE,
    ) -> "DLPredictor":
        """
        Load a model from checkpoint + config JSON + saved vocabulary.

        Args:
            model_cls:         Model class (TextCNN, CNNLSTMClassifier, ...).
            checkpoint_path:   Path to checkpoint_best.pt.
            model_config_path: Path to model_config.json (saved during training).
            model_name:        Display name ("CNN", "CNN-LSTM", ...).
            device:            Target device.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}.\n"
                f"Run the corresponding run_*.py script first."
            )
        if not model_config_path.exists():
            raise FileNotFoundError(
                f"No model config at {model_config_path}."
            )

        with open(model_config_path) as f:
            cfg = json.load(f)

        # Each model class must accept **cfg minus non-init keys
        init_keys = {"vocab_size", "embed_dim", "num_filters", "filter_sizes",
                     "kernel_size", "lstm_hidden", "lstm_layers",
                     "bidirectional", "dropout", "num_classes", "pad_idx"}
        model_kwargs = {k: v for k, v in cfg.items() if k in init_keys}
        model = model_cls(**model_kwargs)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "[DLPredictor] %s loaded (epoch=%d  val_f1=%.4f)",
            model_name, ckpt.get("epoch", -1), ckpt.get("val_f1", 0.0),
        )

        vocab   = Vocabulary.load()
        max_len = cfg.get("max_len", MAX_LEN)
        return cls(model, vocab, model_name, max_len, device)

    # ── BasePredictor interface ────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> dict:
        """
        Batch inference on raw review strings.

        Returns:
            {
              "labels":        list[int],
              "label_names":   list[str],
              "confidences":   list[float],   P(positive)
              "latency_ms":    float,
              "ms_per_sample": float,
            }
        """
        t0   = time.perf_counter()
        seqs = self.vocabulary.encode_batch(texts, self.max_len)   # (N, L)
        x    = torch.tensor(seqs, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()

        labels  = proba.argmax(axis=1).tolist()
        confs   = proba[:, 1].tolist()
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "labels":        labels,
            "label_names":   [LABEL_MAP[l] for l in labels],
            "confidences":   [round(c, 4) for c in confs],
            "latency_ms":    round(elapsed, 3),
            "ms_per_sample": round(elapsed / max(len(texts), 1), 4),
        }

    def predict_single(self, text: str) -> dict:
        """Single-sample inference."""
        result = self.predict([text])
        return {
            "label":      result["labels"][0],
            "label_name": result["label_names"][0],
            "confidence": result["confidences"][0],
            "latency_ms": result["latency_ms"],
        }

    def __repr__(self) -> str:
        return (
            f"DLPredictor(model='{self.model_name}', "
            f"vocab_size={len(self.vocabulary)}, device='{self.device}')"
        )
