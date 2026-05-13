"""
preprocessing/transformer_preprocessor.py
───────────────────────────────────────────
Preprocessing pipeline for BERT and DistilBERT.

Pipeline:
    raw text (globally cleaned)
        → ONLY structural cleaning (HTML already stripped globally)
        → HuggingFace AutoTokenizer
        → {input_ids, attention_mask, token_type_ids} tensors
        → SentimentDataset (PyTorch Dataset)

Cardinal rule: No lowercasing, stemming, stop-word removal, or
punctuation stripping. The tokenizer handles all of this internally.

Artifacts saved:
    preprocessing/artifacts/tokenizers/<model_name>/
        tokenizer_config.json, vocab.txt, special_tokens_map.json, ...
"""

import json
import pathlib
import logging

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from preprocessing.config import (
    TRANSFORMER_MODELS, TRANSFORMER_MAX_LEN, ARTIFACTS_DIR,
)
from preprocessing.global_cleaner import clean as _global_clean

logger = logging.getLogger(__name__)


# ── Transformer-safe cleaning ─────────────────────────────────────────────────

def _minimal_clean(text: str) -> str:
    """
    Apply ONLY global structural cleaning (HTML stripping, whitespace).
    Returns an empty string if cleaning yields None (tokenizer handles it).
    Do NOT apply any further normalization here.
    """
    cleaned = _global_clean(text, min_len=1)
    return cleaned if cleaned is not None else ""


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    PyTorch Dataset wrapping HuggingFace tokenizer output.
    Compatible with torch.utils.data.DataLoader out of the box.

    Works with or without labels (training vs. inference mode).
    """

    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings   # dict of str → (N, seq_len) tensors
        self.labels    = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Transformer Preprocessor ──────────────────────────────────────────────────

class TransformerPreprocessor:
    """
    End-to-end tokenization for BERT and DistilBERT.

    Usage:
        tp = TransformerPreprocessor("bert-base-uncased")
        train_ds = tp.prepare(train_texts, train_labels)
        test_ds  = tp.prepare(test_texts,  test_labels)
        tp.save_tokenizer()

        # Reload for inference:
        tp = TransformerPreprocessor.from_saved("bert-base-uncased")
        ds = tp.prepare(new_texts)
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = TRANSFORMER_MAX_LEN):
        self.model_name = model_name
        self.max_length = max_length
        logger.info("Loading tokenizer: %s …", model_name)
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer ready | vocab_size: %d", self.tokenizer.vocab_size)

    # ── Core method ───────────────────────────────────────────────────────────

    def prepare(
        self,
        texts: list[str],
        labels: list[int] | None = None,
    ) -> SentimentDataset:
        """
        Minimally clean texts, tokenize, and return a SentimentDataset.

        Args:
            texts:  Raw review strings (title + content already merged).
            labels: Integer labels (0=negative, 1=positive). None for inference.

        Returns:
            SentimentDataset ready for DataLoader.
        """
        logger.info("Applying transformer-safe cleaning to %d texts …", len(texts))
        cleaned = [_minimal_clean(t) for t in texts]

        logger.info("Tokenizing with %s (max_length=%d) …", self.model_name, self.max_length)
        encodings = self.tokenizer(
            cleaned,
            truncation            = True,
            padding               = "max_length",
            max_length            = self.max_length,
            return_tensors        = "pt",
            return_attention_mask = True,
        )
        logger.info("Tokenization done | input_ids: %s", tuple(encodings["input_ids"].shape))
        return SentimentDataset(encodings, labels)

    def tokenize_single(self, text: str) -> dict:
        """
        Tokenize one text for single-sample inference.
        Returns a dict of tensors with batch dimension of 1.
        """
        return self.tokenizer(
            _minimal_clean(text),
            truncation            = True,
            padding               = "max_length",
            max_length            = self.max_length,
            return_tensors        = "pt",
            return_attention_mask = True,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _artifact_dir(self) -> pathlib.Path:
        short = self.model_name.replace("/", "_").replace("-", "_")
        d = ARTIFACTS_DIR / "tokenizers" / short
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_tokenizer(self) -> pathlib.Path:
        """
        Save full tokenizer files (vocab, config, special tokens map).
        These allow offline/air-gapped inference without downloading from HF Hub.
        """
        out = self._artifact_dir()
        self.tokenizer.save_pretrained(str(out))
        logger.info("Tokenizer saved → %s", out)
        return out

    def save_config(self) -> pathlib.Path:
        """Save a lightweight JSON config for this preprocessor instance."""
        out = self._artifact_dir() / "preprocessor_config.json"
        config = {
            "model_name":   self.model_name,
            "max_length":   self.max_length,
            "vocab_size":   self.tokenizer.vocab_size,
            "pad_token":    self.tokenizer.pad_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "cls_token":    getattr(self.tokenizer, "cls_token", None),
            "sep_token":    getattr(self.tokenizer, "sep_token", None),
            "do_lower_case": getattr(self.tokenizer, "do_lower_case", None),
        }
        with open(out, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Preprocessor config saved → %s", out)
        return out

    @classmethod
    def from_saved(
        cls,
        model_name: str,
        max_length: int = TRANSFORMER_MAX_LEN,
    ) -> "TransformerPreprocessor":
        """
        Reload a TransformerPreprocessor from the local tokenizer artifact.
        Falls back to HuggingFace Hub if the local artifact does not exist.
        """
        short    = model_name.replace("/", "_").replace("-", "_")
        local    = ARTIFACTS_DIR / "tokenizers" / short
        src      = str(local) if local.exists() else model_name
        instance = cls.__new__(cls)
        instance.model_name = model_name
        instance.max_length = max_length
        instance.tokenizer  = AutoTokenizer.from_pretrained(src)
        logger.info("TransformerPreprocessor loaded from: %s", src)
        return instance


# ── Convenience factory ────────────────────────────────────────────────────────

def build_all_transformer_preprocessors(
    max_length: int = TRANSFORMER_MAX_LEN,
) -> dict[str, TransformerPreprocessor]:
    """
    Build and return preprocessors for all configured transformer models.
    Useful for the run_preprocessing.py orchestration script.
    """
    return {
        key: TransformerPreprocessor(model_name=name, max_length=max_length)
        for key, name in TRANSFORMER_MODELS.items()
    }
