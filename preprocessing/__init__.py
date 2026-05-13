"""
preprocessing/__init__.py
──────────────────────────
Public API for the preprocessing package.
Import from here in training scripts — avoids exposing internal module paths.
"""

from preprocessing.config import (
    SEED,
    SPLITS_DIR,
    ARTIFACTS_DIR,
    TFIDF_CONFIG,
    DL_MAX_LEN,
    DL_EMBED_DIM,
    DL_MIN_FREQ,
    TRANSFORMER_MODELS,
    TRANSFORMER_MAX_LEN,
    LABEL_MAP,
)

from preprocessing.global_cleaner import (
    clean,
    clean_series,
    clean_dataframe,
    strip_html,
    normalize_whitespace,
    is_valid,
)

from preprocessing.classical_preprocessor import ClassicalPreprocessor
from preprocessing.dl_preprocessor import Vocabulary, DLPreprocessor
from preprocessing.transformer_preprocessor import (
    TransformerPreprocessor,
    SentimentDataset,
    build_all_transformer_preprocessors,
)
from preprocessing.artifact_manager import (
    save_joblib,
    save_pickle,
    save_json,
    load_joblib,
    load_pickle,
    load_json,
    save_preprocessing_log,
    list_artifacts,
)

__all__ = [
    # Config
    "SEED", "SPLITS_DIR", "ARTIFACTS_DIR",
    "TFIDF_CONFIG", "DL_MAX_LEN", "DL_EMBED_DIM", "DL_MIN_FREQ",
    "TRANSFORMER_MODELS", "TRANSFORMER_MAX_LEN", "LABEL_MAP",
    # Global cleaning
    "clean", "clean_series", "clean_dataframe",
    "strip_html", "normalize_whitespace", "is_valid",
    # Preprocessors
    "ClassicalPreprocessor",
    "Vocabulary", "DLPreprocessor",
    "TransformerPreprocessor", "SentimentDataset",
    "build_all_transformer_preprocessors",
    # Artifact I/O
    "save_joblib", "save_pickle", "save_json",
    "load_joblib", "load_pickle", "load_json",
    "save_preprocessing_log", "list_artifacts",
]
