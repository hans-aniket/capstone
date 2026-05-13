"""
models/classical/train_classical.py
────────────────────────────────────
Trains three classical ML classifiers on TF-IDF features:
  1. Logistic Regression
  2. Linear SVM (SGDClassifier)
  3. Multinomial Naive Bayes

Saves per-model metrics to results/classical_results.json and
a confusion-matrix figure to results/classical_cm.png.

Usage:
    python models/classical/train_classical.py
"""

import json
import time
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ── Paths ─────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parents[2]
DATA_DIR   = ROOT / "data" / "splits"
RESULTS    = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
print("Loading data …")
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── TF-IDF config ─────────────────────────────────────────────────────
TFIDF = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100_000,
    sublinear_tf=True,
    min_df=2,
)

# ── Model definitions ─────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": Pipeline([
        ("tfidf", TFIDF),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000,
                                  sublinear_tf=True, min_df=2)),
        ("clf",   SGDClassifier(loss="hinge", max_iter=100, random_state=42)),
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000,
                                  sublinear_tf=True, min_df=2)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ]),
}

# ── Training & evaluation ─────────────────────────────────────────────
all_results = {}
fig, axes   = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Classical ML — Confusion Matrices (amazon_polarity)", fontsize=13)

for idx, (name, pipeline) in enumerate(MODELS.items()):
    print(f"\n{'─'*50}")
    print(f"Training: {name}")

    # Train
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Predict
    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    infer_time_ms = (time.perf_counter() - t1) / len(X_test) * 1000

    # Probabilities for ROC-AUC (where available)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline.named_steps["clf"], "decision_function"):
        y_prob = pipeline.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    else:
        y_prob = y_pred.astype(float)

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Train time : {train_time:.1f}s")
    print(f"  Infer (ms/sample): {infer_time_ms:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    all_results[name] = {
        "accuracy":          round(acc, 4),
        "macro_f1":          round(f1, 4),
        "roc_auc":           round(roc_auc, 4),
        "train_time_s":      round(train_time, 2),
        "infer_ms_per_sample": round(infer_time_ms, 4),
    }

    # Confusion matrix subplot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
cm_path = RESULTS / "classical_cm.png"
plt.savefig(cm_path, dpi=150)
print(f"\n📊  Confusion matrix saved → {cm_path}")

# Save JSON results
out_path = RESULTS / "classical_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"📄  Metrics saved        → {out_path}")
