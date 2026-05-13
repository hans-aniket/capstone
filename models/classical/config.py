"""
models/classical/config.py
───────────────────────────
Hyperparameters and output paths for the three classical ML models.
All values reference the project-level preprocessing config where applicable.
"""

import pathlib
import sys

# ── Ensure project root is importable when run directly ───────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.config import ARTIFACTS_DIR, SPLITS_DIR

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR  = _PROJECT_ROOT / "results" / "classical"
MODELS_DIR   = _PROJECT_ROOT / "models" / "classical" / "saved"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model output sub-directories (created on first save) ──────────────────────
def model_dir(model_name: str) -> pathlib.Path:
    """Return (and create) the results directory for a named model."""
    d = RESULTS_DIR / model_name.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d

def saved_model_dir(model_name: str) -> pathlib.Path:
    """Return (and create) the serialized model directory."""
    d = MODELS_DIR / model_name.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Model hyperparameters ─────────────────────────────────────────────────────

NAIVE_BAYES_CONFIG = {
    "alpha": 0.1,        # Laplace smoothing; 0.1 works better than 1.0 with sublinear TF-IDF
    "fit_prior": True,   # Learn class prior probabilities from data
}

LOGISTIC_REGRESSION_CONFIG = {
    "C":         1.0,       # Inverse regularization strength
    "solver":    "lbfgs",   # Efficient for dense, small problems; saga for large sparse
    "max_iter":  1000,
    "n_jobs":    -1,        # Use all CPU cores
    "random_state": 42,
}

SVM_CONFIG = {
    # LinearSVC is fast; CalibratedClassifierCV wraps it for predict_proba
    "svc_C":         1.0,
    "svc_max_iter":  1000,
    "calibration_cv":     3,       # 3-fold calibration — fast on 20k samples
    "calibration_method": "sigmoid",
}

# ── Registry: name → config (used by run_training.py) ────────────────────────
MODEL_REGISTRY = {
    "naive_bayes":          NAIVE_BAYES_CONFIG,
    "logistic_regression":  LOGISTIC_REGRESSION_CONFIG,
    "svm":                  SVM_CONFIG,
}

DISPLAY_NAMES = {
    "naive_bayes":         "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "svm":                 "SVM (Linear)",
}
