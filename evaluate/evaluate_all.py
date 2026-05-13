"""
evaluate/evaluate_all.py
─────────────────────────
Aggregates all results JSON files from results/ and produces:
  1. A summary table printed to stdout
  2. results/comparison_bar.png  — bar chart (Accuracy, F1, AUC)
  3. results/latency_bar.png     — inference latency comparison
  4. results/summary_table.csv   — machine-readable results

Run AFTER all model training scripts have completed:
    python evaluate/evaluate_all.py
"""

import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).parents[1]
RESULTS = ROOT / "results"

# ── Model display names (key = JSON filename stem) ────────────────────
MODEL_LABELS = {
    "classical_results":           None,        # handled specially (nested)
    "bilstm_results":              "BiLSTM",
    "textcnn_results":             "TextCNN",
    "distilbert_base_uncased_results": "DistilBERT",
    "bert_base_uncased_results":   "BERT-base",
    "roberta_base_results":        "RoBERTa-base",
}

TIER_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Linear SVM":          "#4C72B0",
    "Naive Bayes":         "#4C72B0",
    "BiLSTM":              "#DD8452",
    "TextCNN":             "#DD8452",
    "DistilBERT":          "#55A868",
    "BERT-base":           "#55A868",
    "RoBERTa-base":        "#55A868",
}

TIER_HATCHES = {
    "Logistic Regression": "",
    "Linear SVM":          "//",
    "Naive Bayes":         "xx",
    "BiLSTM":              "",
    "TextCNN":             "//",
    "DistilBERT":          "",
    "BERT-base":           "//",
    "RoBERTa-base":        "xx",
}


def load_all_results() -> list[dict]:
    rows = []

    for json_path in sorted(RESULTS.glob("*.json")):
        stem = json_path.stem

        # Classical results are nested dicts
        if stem == "classical_results":
            data = json.loads(json_path.read_text())
            for model_name, metrics in data.items():
                rows.append({"model": model_name, **metrics})
            continue

        label = MODEL_LABELS.get(stem)
        if label is None:
            continue

        data = json.loads(json_path.read_text())
        rows.append({"model": label, **data})

    return rows


def main():
    rows = load_all_results()
    if not rows:
        print("❌  No result files found in results/. Run all training scripts first.")
        return

    df = pd.DataFrame(rows)
    df = df[["model", "accuracy", "macro_f1", "roc_auc", "infer_ms_per_sample"]]
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\n" + "="*70)
    print(" COMPARATIVE SENTIMENT ANALYSIS RESULTS — amazon_polarity")
    print("="*70)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("="*70 + "\n")

    df.to_csv(RESULTS / "summary_table.csv", index=False)
    print(f"📄  Table saved → {RESULTS / 'summary_table.csv'}")

    models = df["model"].tolist()
    colors  = [TIER_COLORS.get(m, "#999999") for m in models]
    hatches = [TIER_HATCHES.get(m, "") for m in models]
    x = np.arange(len(models))

    # ── Bar chart: Accuracy / F1 / AUC ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparative Sentiment Analysis — amazon_polarity", fontsize=13, weight="bold")

    for ax, metric, ylabel in zip(
        axes,
        ["accuracy", "macro_f1", "roc_auc"],
        ["Accuracy", "Macro F1", "ROC-AUC"],
    ):
        bars = ax.bar(x, df[metric], color=colors, edgecolor="white", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(max(0, df[metric].min() - 0.05), min(1.0, df[metric].max() + 0.05))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Legend: tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Classical ML"),
        Patch(facecolor="#DD8452", label="Deep Learning"),
        Patch(facecolor="#55A868", label="Transformers"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    bar_path = RESULTS / "comparison_bar.png"
    plt.savefig(bar_path, dpi=150)
    print(f"📊  Bar chart saved     → {bar_path}")

    # ── Latency chart ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    lat_bars = ax2.barh(models[::-1], df["infer_ms_per_sample"][::-1],
                        color=[colors[i] for i in range(len(models)-1, -1, -1)])
    ax2.set_xlabel("Inference latency (ms / sample)")
    ax2.set_title("Inference Speed Comparison — amazon_polarity", weight="bold")
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(lat_bars, df["infer_ms_per_sample"][::-1]):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f} ms", va="center", fontsize=8)
    plt.tight_layout()
    lat_path = RESULTS / "latency_bar.png"
    plt.savefig(lat_path, dpi=150)
    print(f"📊  Latency chart saved → {lat_path}")

    plt.show()


if __name__ == "__main__":
    main()
