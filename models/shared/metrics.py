"""
models/shared/metrics.py
─────────────────────────
Shared evaluation utilities used by all model tiers:
  Classical ML, LSTM, CNN-LSTM, BERT, DistilBERT.

All training pipelines call these functions to produce consistent,
comparable metric reports.
"""

import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

LABEL_NAMES = ["negative", "positive"]


# ── Core metric computation ────────────────────────────────────────────────────

def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: list | np.ndarray | None = None,
    label_names: list[str] = LABEL_NAMES,
) -> dict:
    """
    Compute the full metric suite for binary sentiment classification.

    Args:
        y_true:     Ground-truth integer labels.
        y_pred:     Predicted integer labels.
        y_prob:     Predicted positive-class probabilities (for ROC-AUC).
                    If None, ROC-AUC is omitted.
        label_names: Human-readable class names.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc (if y_prob given).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro":   round(float(precision_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
        "precision_weighted":round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro":      round(float(recall_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "recall_weighted":   round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro":          round(float(f1_score(y_true, y_pred, average="macro",        zero_division=0)), 4),
        "f1_weighted":       round(float(f1_score(y_true, y_pred, average="weighted",     zero_division=0)), 4),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    return metrics


def add_timing(metrics: dict, train_time_s: float, infer_time_s: float, n_samples: int) -> dict:
    """Attach timing information to a metrics dict."""
    metrics["train_time_s"]      = round(train_time_s, 3)
    metrics["infer_time_s"]      = round(infer_time_s, 6)
    metrics["infer_ms_per_sample"] = round(infer_time_s / max(n_samples, 1) * 1000, 4)
    return metrics


# ── Text report ────────────────────────────────────────────────────────────────

def generate_classification_report(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    label_names: list[str] = LABEL_NAMES,
) -> str:
    """Return sklearn's classification_report as a formatted string."""
    return classification_report(y_true, y_pred, target_names=label_names, digits=4)


def save_classification_report(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    path: pathlib.Path,
    label_names: list[str] = LABEL_NAMES,
    header: str = "",
) -> pathlib.Path:
    """Write the classification report to a .txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_classification_report(y_true, y_pred, label_names)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n" + "-" * len(header) + "\n\n")
        f.write(report)
    return path


# ── Metrics JSON ───────────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: pathlib.Path) -> pathlib.Path:
    """Save a metrics dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics_json(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    path: pathlib.Path,
    label_names: list[str] = LABEL_NAMES,
    title: str = "Confusion Matrix",
) -> pathlib.Path:
    """
    Plot and save a confusion matrix as a PNG.
    Returns the path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title,        fontsize=12, weight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Convenience: save everything at once ──────────────────────────────────────

def save_all_evaluation_artifacts(
    y_true:       list | np.ndarray,
    y_pred:       list | np.ndarray,
    y_prob:       list | np.ndarray | None,
    metrics:      dict,
    output_dir:   pathlib.Path,
    model_name:   str,
) -> None:
    """
    One-call helper: saves metrics.json, classification_report.txt,
    and confusion_matrix.png into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / "metrics.json")
    save_classification_report(
        y_true, y_pred,
        output_dir / "classification_report.txt",
        header=f"Model: {model_name}",
    )
    save_confusion_matrix(
        y_true, y_pred,
        output_dir / "confusion_matrix.png",
        title=f"{model_name} — Confusion Matrix",
    )
