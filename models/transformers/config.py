"""
models/transformers/config.py
──────────────────────────────
Hyperparameters and paths for BERT and DistilBERT models.
"""

import pathlib
import sys

import torch

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import TRANSFORMER_MODELS, TRANSFORMER_MAX_LEN

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES = 2

TRAIN_CONFIG = {
    "batch_size":     16,       # Lower batch size due to memory constraints
    "num_epochs":     3,        # Transformers fine-tune quickly (2-4 epochs standard)
    "lr":             2e-5,     # Standard learning rate for fine-tuning
    "weight_decay":   0.01,
    "patience":       2,        # Early stopping patience
    "clip_grad_norm": 1.0,
    "val_split":      0.1,      # 10% of train data for validation
    "seed":           42,
    "warmup_steps":   0,        # Optional: num warmup steps for scheduler
}

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_BASE = _PROJECT_ROOT / "results" / "transformers"
SAVED_BASE   = _PROJECT_ROOT / "models" / "transformers" / "saved"

def get_model_paths(model_key: str):
    """
    Returns paths for a given transformer model (e.g. 'bert', 'distilbert').
    Creates directories if they don't exist.
    """
    results_dir = RESULTS_BASE / model_key
    saved_dir   = SAVED_BASE / model_key
    
    results_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "results_dir":  results_dir,
        "saved_dir":    saved_dir,
        "ckpt_best":    saved_dir / "checkpoint_best.pt",
        "ckpt_last":    saved_dir / "checkpoint_last.pt",
        "history":      saved_dir / "training_history.json",
        "model_config": saved_dir / "model_config.json"
    }
