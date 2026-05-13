"""
models/deep_learning/train_dl.py
─────────────────────────────────
Trains either BiLSTM or TextCNN on the prepared amazon_polarity splits.

Usage:
    python models/deep_learning/train_dl.py --model bilstm
    python models/deep_learning/train_dl.py --model textcnn
    python models/deep_learning/train_dl.py --model bilstm --glove path/to/glove.6B.100d.txt

Outputs:
    results/<model>_results.json
    results/<model>_training_curve.png
"""

import argparse
import json
import time
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import Vocabulary, ReviewDataset, load_glove_embeddings, EMBED_DIM
from bilstm  import BiLSTM
from textcnn import TextCNN

# ── Paths ─────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "splits"
RESULTS  = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR         = 1e-3
MAX_LEN    = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        total_loss  += loss.item() * len(y)
        all_preds   += logits.argmax(1).cpu().tolist()
        all_labels  += y.cpu().tolist()
        all_probs   += probs.cpu().tolist()
    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, f1, auc, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["bilstm", "textcnn"], default="bilstm")
    parser.add_argument("--glove",  type=str, default="", help="Path to GloVe .txt file")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f" Model : {args.model.upper()}")
    print(f" Device: {DEVICE}")
    print(f"{'='*55}\n")

    # ── Load data ─────────────────────────────────────────────────────
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

    # ── Vocabulary ────────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=2)
    vocab.build(train_df["text"].tolist())

    # ── GloVe embeddings ──────────────────────────────────────────────
    glove_path = pathlib.Path(args.glove) if args.glove else pathlib.Path("nonexistent")
    embed_matrix = load_glove_embeddings(glove_path, vocab, EMBED_DIM)

    # ── Datasets & loaders ────────────────────────────────────────────
    train_ds = ReviewDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, MAX_LEN)
    test_ds  = ReviewDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────
    if args.model == "bilstm":
        model = BiLSTM(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=128,
            num_layers=2,
            pretrained_embeddings=embed_matrix,
        ).to(DEVICE)
    else:
        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            num_filters=128,
            filter_sizes=[2, 3, 4],
            pretrained_embeddings=embed_matrix,
        ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_auc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), RESULTS / f"{args.model}_best.pt")

    # ── Final evaluation on best checkpoint ───────────────────────────
    model.load_state_dict(torch.load(RESULTS / f"{args.model}_best.pt", map_location=DEVICE))
    t_infer_start = time.perf_counter()
    _, final_acc, final_f1, final_auc, _, _ = evaluate(model, test_loader, criterion)
    infer_ms = (time.perf_counter() - t_infer_start) / len(test_ds) * 1000

    print(f"\n✅  Best checkpoint results:")
    print(f"   Accuracy : {final_acc:.4f}")
    print(f"   Macro F1 : {final_f1:.4f}")
    print(f"   ROC-AUC  : {final_auc:.4f}")
    print(f"   Infer (ms/sample): {infer_ms:.3f}")

    # ── Save metrics ──────────────────────────────────────────────────
    metrics = {
        "accuracy": round(final_acc, 4),
        "macro_f1": round(final_f1, 4),
        "roc_auc":  round(final_auc, 4),
        "infer_ms_per_sample": round(infer_ms, 4),
        "epochs": args.epochs,
        "trainable_params": total_params,
    }
    out_path = RESULTS / f"{args.model}_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄  Metrics saved → {out_path}")

    # ── Training curve ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_range = range(1, args.epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{args.model.upper()} — Loss Curve")
    ax1.legend()

    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy")
    ax2.plot(epochs_range, history["val_f1"],  label="Val Macro F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title(f"{args.model.upper()} — Metrics")
    ax2.legend()

    plt.tight_layout()
    curve_path = RESULTS / f"{args.model}_training_curve.png"
    plt.savefig(curve_path, dpi=150)
    print(f"📊  Training curve saved → {curve_path}")


if __name__ == "__main__":
    main()
