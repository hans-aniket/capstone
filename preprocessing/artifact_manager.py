"""
preprocessing/artifact_manager.py
───────────────────────────────────
Unified save/load for all preprocessing artifacts.
Dispatches to joblib (sklearn objects), pickle (vocab), or JSON (configs).
"""

import json
import pickle
import pathlib
import logging
from datetime import datetime, timezone
from typing import Any

import joblib

from preprocessing.config import ARTIFACTS_DIR, ARTIFACT_PREP_LOG

logger = logging.getLogger(__name__)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_joblib(obj: Any, path: pathlib.Path) -> pathlib.Path:
    """Save a sklearn-compatible object with joblib (efficient for sparse matrices)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved (joblib) → %s", path)
    return path


def save_pickle(obj: Any, path: pathlib.Path) -> pathlib.Path:
    """Save an arbitrary Python object with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved (pickle) → %s", path)
    return path


def save_json(data: dict, path: pathlib.Path) -> pathlib.Path:
    """Save a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved (json)   → %s", path)
    return path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_joblib(path: pathlib.Path) -> Any:
    _require_exists(path)
    obj = joblib.load(path)
    logger.info("Loaded (joblib) ← %s", path)
    return obj


def load_pickle(path: pathlib.Path) -> Any:
    _require_exists(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded (pickle) ← %s", path)
    return obj


def load_json(path: pathlib.Path) -> dict:
    _require_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded (json)   ← %s", path)
    return data


# ── Metadata / log ────────────────────────────────────────────────────────────

def save_preprocessing_log(entries: dict) -> pathlib.Path:
    """
    Append a timestamped entry to the preprocessing run log.
    Useful for tracking when each artifact was created and with what config.
    """
    log_path = ARTIFACTS_DIR / ARTIFACT_PREP_LOG
    existing: list = []
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)

    entries["timestamp"] = datetime.now(timezone.utc).isoformat()
    existing.append(entries)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Preprocessing log updated → %s", log_path)
    return log_path


def list_artifacts() -> list[pathlib.Path]:
    """Return all files currently saved in the artifacts directory."""
    files = sorted(ARTIFACTS_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]
    if files:
        logger.info("Artifacts in %s:", ARTIFACTS_DIR)
        for f in files:
            logger.info("  %s  (%d bytes)", f.relative_to(ARTIFACTS_DIR), f.stat().st_size)
    else:
        logger.info("No artifacts found in %s", ARTIFACTS_DIR)
    return files


# ── Internal ──────────────────────────────────────────────────────────────────

def _require_exists(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Run run_preprocessing.py first to generate it."
        )
