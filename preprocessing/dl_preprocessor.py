"""
preprocessing/dl_preprocessor.py
──────────────────────────────────
Preprocessing pipeline for LSTM and CNN-LSTM.

Pipeline:
    raw text (globally cleaned)
        → moderate normalization (lowercase, keep ! ? . , ')
        → whitespace tokenization
        → Vocabulary lookup (fit on train only)
        → integer index sequences
        → right-pad / right-truncate to DL_MAX_LEN

Artifacts saved:
    preprocessing/artifacts/vocabulary.pkl
    preprocessing/artifacts/vocab_config.json
"""

import re
import json
import pickle
import pathlib
import logging
from collections import Counter

import numpy as np

from preprocessing.config import (
    DL_MAX_LEN, DL_MIN_FREQ, DL_EMBED_DIM,
    PAD_TOKEN, UNK_TOKEN,
    ARTIFACTS_DIR, ARTIFACT_VOCABULARY, ARTIFACT_VOCAB_CFG,
)
from preprocessing.global_cleaner import clean
from preprocessing.artifact_manager import save_pickle, load_pickle, save_json

logger = logging.getLogger(__name__)

# ── DL-specific normalization ─────────────────────────────────────────────────
_SPECIAL_RE = re.compile(r"[^a-z0-9\s!?.,']")
_WS_RE      = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """
    Moderate normalization for deep learning.
    - Lowercase (reduces vocabulary size)
    - Remove special chars except core sentiment punctuation (! ? . , ')
    - Normalize whitespace
    Applied AFTER global cleaning.
    """
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Word-to-index mapping built exclusively from training data.

    Token index convention:
        0  →  <PAD>  (padding; embedding row is zeroed, no gradient)
        1  →  <UNK>  (out-of-vocabulary at inference time)
        2+ →  regular tokens sorted by frequency (descending)
    """

    def __init__(self, min_freq: int = DL_MIN_FREQ):
        self.min_freq   = min_freq
        self.token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token: list[str]      = [PAD_TOKEN, UNK_TOKEN]
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, texts: list[str]) -> None:
        """
        Build vocabulary from training texts only.
        ⚠️  Never call on test data.
        """
        logger.info("Building vocabulary from %d texts (min_freq=%d) …",
                    len(texts), self.min_freq)
        counter: Counter = Counter()
        for text in texts:
            cleaned = clean(text)
            if cleaned:
                counter.update(_normalize(cleaned).split())

        for token, freq in counter.most_common():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.idx2token)
                self.idx2token.append(token)

        self._built = True
        logger.info("Vocabulary built | size: %d  (raw tokens seen: %d)",
                    len(self), len(counter))

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str, max_len: int = DL_MAX_LEN) -> list[int]:
        """
        Encode one text → padded integer list of length max_len.
        - OOV tokens → index 1 (<UNK>)
        - Short sequences → right-padded with 0 (<PAD>)
        - Long sequences  → right-truncated at max_len
        """
        cleaned = clean(text)
        if cleaned is None:
            return [0] * max_len
        tokens = _normalize(cleaned).split()[:max_len]
        ids    = [self.token2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))   # right-pad
        return ids

    def encode_batch(self, texts: list[str], max_len: int = DL_MAX_LEN) -> np.ndarray:
        """Encode a list of texts → int32 array of shape (N, max_len)."""
        return np.array([self.encode(t, max_len) for t in texts], dtype=np.int32)

    def decode(self, ids: list[int]) -> str:
        """Convert integer ids back to tokens (skips PAD). For debugging."""
        return " ".join(self.idx2token[i] for i in ids if i != 0)

    # ── Length statistics (call before fixing DL_MAX_LEN) ─────────────────────

    def compute_length_stats(self, texts: list[str]) -> dict:
        """
        Compute token count percentiles on training texts.
        Use these to validate or tune DL_MAX_LEN in config.py.
        """
        lengths = []
        for text in texts:
            cleaned = clean(text)
            lengths.append(len(_normalize(cleaned).split()) if cleaned else 0)
        arr = np.array(lengths)
        stats = {
            "min":  int(arr.min()),
            "p50":  int(np.percentile(arr, 50)),
            "p90":  int(np.percentile(arr, 90)),
            "p95":  int(np.percentile(arr, 95)),
            "p99":  int(np.percentile(arr, 99)),
            "max":  int(arr.max()),
            "mean": round(float(arr.mean()), 1),
        }
        logger.info("Token length stats: %s", stats)
        return stats

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: pathlib.Path | None = None) -> pathlib.Path:
        out = path or (ARTIFACTS_DIR / ARTIFACT_VOCABULARY)
        return save_pickle(self, out)

    def save_config(self, path: pathlib.Path | None = None) -> pathlib.Path:
        out = path or (ARTIFACTS_DIR / ARTIFACT_VOCAB_CFG)
        return save_json(
            {
                "vocab_size":  len(self),
                "min_freq":    self.min_freq,
                "pad_token":   PAD_TOKEN,
                "pad_index":   0,
                "unk_token":   UNK_TOKEN,
                "unk_index":   1,
                "max_len":     DL_MAX_LEN,
                "embed_dim":   DL_EMBED_DIM,
            },
            out,
        )

    @classmethod
    def load(cls, path: pathlib.Path | None = None) -> "Vocabulary":
        src      = path or (ARTIFACTS_DIR / ARTIFACT_VOCABULARY)
        instance = load_pickle(src)
        logger.info("Vocabulary loaded | size: %d", len(instance))
        return instance

    def __len__(self) -> int:
        return len(self.idx2token)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx


# ── DLPreprocessor ────────────────────────────────────────────────────────────

class DLPreprocessor:
    """
    End-to-end preprocessing for LSTM and CNN-LSTM.

    Usage:
        dp = DLPreprocessor()
        X_train = dp.fit_transform(train_texts)  # builds vocab, encodes
        X_test  = dp.transform(test_texts)        # encodes with saved vocab
        dp.save()

        # Reload for inference:
        dp = DLPreprocessor.load()
        X_new = dp.transform(new_texts)
    """

    def __init__(self, min_freq: int = DL_MIN_FREQ, max_len: int = DL_MAX_LEN):
        self.max_len    = max_len
        self.vocabulary = Vocabulary(min_freq=min_freq)

    def fit_transform(self, train_texts: list[str]) -> np.ndarray:
        """Build vocabulary from train, then encode. ⚠️  Train only."""
        self.vocabulary.build(train_texts)
        logger.info("Encoding %d training texts → shape (%d, %d) …",
                    len(train_texts), len(train_texts), self.max_len)
        return self.vocabulary.encode_batch(train_texts, self.max_len)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the fitted vocabulary."""
        self._assert_fitted()
        return self.vocabulary.encode_batch(texts, self.max_len)

    def length_stats(self, texts: list[str]) -> dict:
        """Delegate to vocabulary's length statistics method."""
        return self.vocabulary.compute_length_stats(texts)

    def save(self) -> None:
        """Save vocabulary pickle and human-readable config JSON."""
        self.vocabulary.save()
        self.vocabulary.save_config()

    @classmethod
    def load(cls, max_len: int = DL_MAX_LEN) -> "DLPreprocessor":
        instance          = cls.__new__(cls)
        instance.max_len  = max_len
        instance.vocabulary = Vocabulary.load()
        return instance

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def _assert_fitted(self) -> None:
        if not self.vocabulary._built:
            raise RuntimeError(
                "DLPreprocessor vocabulary is not built. "
                "Call fit_transform(train_texts) first."
            )
