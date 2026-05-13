"""
models/deep_learning/config.py
───────────────────────────────
Hyperparameters and paths for all deep learning models.
LSTM-specific values are here; CNN-LSTM will extend this file later.
"""

import pathlib
import sys

import torch

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Shared DL constants (mirrors preprocessing/config.py) ────────────────────
MAX_LEN    = 200    # must match DL_MAX_LEN in preprocessing/config.py
EMBED_DIM  = 100    # must match DL_EMBED_DIM
PAD_IDX    = 0      # <PAD> token index — embedding row zeroed, no gradient
NUM_CLASSES = 2

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_CONFIG = {
    "hidden_dim":    256,
    "num_layers":    2,
    "dropout":       0.3,
    "bidirectional": True,
}

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size":     64,
    "num_epochs":     10,
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "patience":       3,        # early stopping patience (epochs with no val_f1 gain)
    "clip_grad_norm": 1.0,
    "val_split":      0.1,      # fraction of train data used for validation
    "seed":           42,
}

# ── GloVe embeddings (optional) ───────────────────────────────────────────────
# Set to a valid path if you have glove.6B.100d.txt downloaded.
# If the path does not exist, random embeddings are used instead.
GLOVE_PATH = _PROJECT_ROOT / "embeddings" / "glove.6B.100d.txt"

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_BASE = _PROJECT_ROOT / "results" / "deep_learning"
SAVED_BASE   = _PROJECT_ROOT / "models" / "deep_learning" / "saved"

LSTM_RESULTS_DIR = RESULTS_BASE / "lstm"
LSTM_SAVED_DIR   = SAVED_BASE   / "lstm"

LSTM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LSTM_SAVED_DIR.mkdir(parents=True, exist_ok=True)

LSTM_CHECKPOINT_BEST   = LSTM_SAVED_DIR / "checkpoint_best.pt"
LSTM_CHECKPOINT_LAST   = LSTM_SAVED_DIR / "checkpoint_last.pt"
LSTM_TRAINING_HISTORY  = LSTM_SAVED_DIR / "training_history.json"
LSTM_MODEL_CONFIG_FILE = LSTM_SAVED_DIR / "model_config.json"

# ── TextCNN architecture ───────────────────────────────────────────────────────
CNN_CONFIG = {
    "num_filters":  128,        # filters per kernel size
    "filter_sizes": [2, 3, 4],  # parallel kernel widths (bigrams, trigrams, 4-grams)
    "dropout":      0.5,
}

CNN_RESULTS_DIR = RESULTS_BASE / "cnn"
CNN_SAVED_DIR   = SAVED_BASE   / "cnn"

CNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CNN_SAVED_DIR.mkdir(parents=True, exist_ok=True)

CNN_CHECKPOINT_BEST   = CNN_SAVED_DIR / "checkpoint_best.pt"
CNN_CHECKPOINT_LAST   = CNN_SAVED_DIR / "checkpoint_last.pt"
CNN_TRAINING_HISTORY  = CNN_SAVED_DIR / "training_history.json"
CNN_MODEL_CONFIG_FILE = CNN_SAVED_DIR / "model_config.json"

# ── CNN-LSTM architecture ─────────────────────────────────────────────────────
# CNN extracts local n-gram features at every position;
# BiLSTM then models the sequence of those features.
CNN_LSTM_CONFIG = {
    "num_filters":   128,   # Conv1d output channels
    "kernel_size":   3,     # trigram-level feature extraction
    "lstm_hidden":   128,   # LSTM hidden size (per direction)
    "lstm_layers":   1,
    "bidirectional": True,
    "dropout":       0.3,
}

CNN_LSTM_RESULTS_DIR = RESULTS_BASE / "cnn_lstm"
CNN_LSTM_SAVED_DIR   = SAVED_BASE   / "cnn_lstm"

CNN_LSTM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CNN_LSTM_SAVED_DIR.mkdir(parents=True, exist_ok=True)

CNN_LSTM_CHECKPOINT_BEST   = CNN_LSTM_SAVED_DIR / "checkpoint_best.pt"
CNN_LSTM_CHECKPOINT_LAST   = CNN_LSTM_SAVED_DIR / "checkpoint_last.pt"
CNN_LSTM_TRAINING_HISTORY  = CNN_LSTM_SAVED_DIR / "training_history.json"
CNN_LSTM_MODEL_CONFIG_FILE = CNN_LSTM_SAVED_DIR / "model_config.json"

