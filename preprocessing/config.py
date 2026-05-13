"""
preprocessing/config.py
────────────────────────
Central configuration for all preprocessing pipelines.
All hyperparameters live here — never hardcoded in individual modules.
"""

import pathlib

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).parents[1]
DATA_DIR      = ROOT / "data"
SPLITS_DIR    = DATA_DIR / "splits"
ARTIFACTS_DIR = ROOT / "preprocessing" / "artifacts"

# Created on import so all modules can write immediately
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tokenizers").mkdir(exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_NAME = "amazon_polarity"
LABEL_MAP    = {0: "negative", 1: "positive"}

# ── Global cleaning ────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 10  # characters; texts shorter than this are dropped

# ── Classical ML (TF-IDF) ─────────────────────────────────────────────────────
TFIDF_CONFIG = {
    "ngram_range":   (1, 2),    # unigrams + bigrams → captures "not good", "very bad"
    "max_features":  100_000,   # top-N features by corpus TF-IDF score
    "sublinear_tf":  True,      # log(1+tf) — dampens high-frequency dominance
    "min_df":        2,         # drop hapax legomena
    "max_df":        0.95,      # soft stop-word filter (data-driven)
    "strip_accents": "unicode",
    "analyzer":      "word",
}

# ── Deep Learning (LSTM / CNN-LSTM) ───────────────────────────────────────────
DL_MAX_LEN  = 200   # token sequence length (covers ~95th percentile of reviews)
DL_EMBED_DIM = 100  # must match GloVe file (glove.6B.100d)
DL_MIN_FREQ  = 2    # minimum token frequency for vocabulary inclusion

PAD_TOKEN = "<PAD>"  # index 0 — zero-padded, no gradient
UNK_TOKEN = "<UNK>"  # index 1 — out-of-vocabulary at test time

# ── Transformers (BERT / DistilBERT) ──────────────────────────────────────────
TRANSFORMER_MODELS = {
    "bert":       "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}
TRANSFORMER_MAX_LEN = 128  # subword tokens; covers ~90th percentile of reviews

# ── Artifact filenames ─────────────────────────────────────────────────────────
ARTIFACT_TFIDF       = "tfidf_vectorizer.joblib"
ARTIFACT_VOCABULARY  = "vocabulary.pkl"
ARTIFACT_VOCAB_CFG   = "vocab_config.json"
ARTIFACT_SPLITS_META = "splits_metadata.json"
ARTIFACT_PREP_LOG    = "preprocessing_log.json"
