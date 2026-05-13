"""
preprocessing/classical_preprocessor.py
─────────────────────────────────────────
Preprocessing pipeline for Naive Bayes, Logistic Regression, and SVM.

Pipeline:
    raw text (globally cleaned)
        → aggressive normalization (lowercase, remove punct, numbers)
        → TF-IDF vectorization (fit on train only)
        → sparse matrix output

Fit-on-train-only is enforced by the class interface.
"""

import re
import pathlib
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from preprocessing.config import TFIDF_CONFIG, ARTIFACTS_DIR, ARTIFACT_TFIDF
from preprocessing.global_cleaner import clean
from preprocessing.artifact_manager import save_joblib, load_joblib

logger = logging.getLogger(__name__)

# ── Classical-ML-specific normalization patterns ───────────────────────────────
_PUNCT_RE  = re.compile(r"[^a-z0-9\s]")
_NUM_RE    = re.compile(r"\b\d+\b")
_WS_RE     = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """
    Aggressive normalization for classical ML.
    Applied AFTER global cleaning.
    - Lowercase (vocabulary is case-sensitive without it)
    - Replace isolated numbers with <NUM> token (reduce sparsity)
    - Strip punctuation (bag-of-words can't use it)
    """
    text = text.lower()
    text = _NUM_RE.sub(" <NUM> ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


class ClassicalPreprocessor:
    """
    TF-IDF preprocessing for classical ML models.

    Usage:
        cp = ClassicalPreprocessor()
        X_train = cp.fit_transform(train_texts)   # fit + transform
        X_test  = cp.transform(test_texts)         # transform only — no fitting!
        cp.save()

        # Reload for inference:
        cp = ClassicalPreprocessor.load()
        X_new = cp.transform(new_texts)
    """

    def __init__(self, config: dict | None = None):
        self.vectorizer  = TfidfVectorizer(**(config or TFIDF_CONFIG))
        self._is_fitted  = False

    # ── Core methods ──────────────────────────────────────────────────────────

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Global clean + classical normalization for a list of texts."""
        result = []
        for t in texts:
            cleaned = clean(t)
            result.append(_normalize(cleaned) if cleaned else "")
        return result

    def fit_transform(self, train_texts: list[str]) -> spmatrix:
        """
        Fit TF-IDF on training data and return the transformed matrix.

        ⚠️  NEVER call this with test data — that is data leakage.
        """
        logger.info("Normalizing %d training texts …", len(train_texts))
        normed = self.normalize_batch(train_texts)

        logger.info("Fitting TF-IDF vectorizer …")
        X = self.vectorizer.fit_transform(normed)
        self._is_fitted = True

        logger.info(
            "TF-IDF fitted | vocab: %d  matrix: %s",
            len(self.vectorizer.vocabulary_), X.shape,
        )
        return X

    def transform(self, texts: list[str]) -> spmatrix:
        """Transform texts using the already-fitted vectorizer."""
        self._assert_fitted()
        normed = self.normalize_batch(texts)
        return self.vectorizer.transform(normed)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: pathlib.Path | None = None) -> pathlib.Path:
        self._assert_fitted()
        out = path or (ARTIFACTS_DIR / ARTIFACT_TFIDF)
        return save_joblib(self.vectorizer, out)

    @classmethod
    def load(cls, path: pathlib.Path | None = None) -> "ClassicalPreprocessor":
        src      = path or (ARTIFACTS_DIR / ARTIFACT_TFIDF)
        instance = cls.__new__(cls)
        instance.vectorizer = load_joblib(src)
        instance._is_fitted = True
        return instance

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._is_fitted else 0

    @property
    def feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist() if self._is_fitted else []

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ClassicalPreprocessor is not fitted. "
                "Call fit_transform(train_texts) first."
            )
