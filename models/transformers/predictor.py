"""
models/transformers/predictor.py
──────────────────────────────────
TransformerPredictor — implements BasePredictor for HuggingFace models.
"""

import time
import logging
import pathlib

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

from models.base import BasePredictor
from models.transformers.config import DEVICE
from preprocessing.transformer_preprocessor import TransformerPreprocessor
from preprocessing.config import LABEL_MAP

logger = logging.getLogger(__name__)

class TransformerPredictor(BasePredictor):
    tier: str = "transformers"

    def __init__(
        self,
        model:        torch.nn.Module,
        preprocessor: TransformerPreprocessor,
        model_name:   str,
        device:       torch.device = DEVICE,
    ):
        self.model        = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.model_name   = model_name
        self.device       = device

    @classmethod
    def from_saved(
        cls,
        hf_model_name:     str,
        checkpoint_path:   pathlib.Path,
        model_name:        str,
        device:            torch.device = DEVICE,
    ) -> "TransformerPredictor":
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

        # Load preprocessor from saved offline artifact
        preprocessor = TransformerPreprocessor.from_saved(hf_model_name)

        # Initialize model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name, num_labels=2
        )
        
        # Load custom fine-tuned weights
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
        logger.info(
            "[TransformerPredictor] %s loaded (epoch=%d  val_f1=%.4f)",
            model_name, ckpt.get("epoch", -1), ckpt.get("val_f1", 0.0),
        )

        return cls(model, preprocessor, model_name, device)

    def predict(self, texts: list[str]) -> dict:
        t0 = time.perf_counter()
        
        # Preprocessor returns a SentimentDataset which we can slice
        ds = self.preprocessor.prepare(texts)
        
        all_preds, all_confs = [], []
        
        # Run inference in smaller batches to avoid OOM
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(ds), batch_size):
                end = min(i + batch_size, len(ds))
                batch_texts = texts[i:end]
                
                # We can also just use the preprocessor directly
                encodings = self.preprocessor.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.preprocessor.max_length,
                    return_tensors="pt"
                )
                
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                kwargs = {}
                if "token_type_ids" in encodings:
                    kwargs["token_type_ids"] = encodings["token_type_ids"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
                proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
                all_preds.extend(proba.argmax(axis=1).tolist())
                all_confs.extend(proba[:, 1].tolist())

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "labels":        all_preds,
            "label_names":   [LABEL_MAP[l] for l in all_preds],
            "confidences":   [round(c, 4) for c in all_confs],
            "latency_ms":    round(elapsed, 3),
            "ms_per_sample": round(elapsed / max(len(texts), 1), 4),
        }

    def predict_single(self, text: str) -> dict:
        result = self.predict([text])
        return {
            "label":      result["labels"][0],
            "label_name": result["label_names"][0],
            "confidence": result["confidences"][0],
            "latency_ms": result["latency_ms"],
        }
