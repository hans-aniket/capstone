"""
models/transformers/train_transformer.py
─────────────────────────────────────────
Fine-tunes a HuggingFace transformer on amazon_polarity splits.
Supports DistilBERT, BERT-base, and RoBERTa-base out of the box.

Usage:
    python models/transformers/train_transformer.py --model distilbert-base-uncased
    python models/transformers/train_transformer.py --model bert-base-uncased
    python models/transformers/train_transformer.py --model roberta-base

Outputs:
    results/<short_model_name>_results.json
    results/<short_model_name>_training_curve.png
"""

import argparse
import json
import time
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "splits"
RESULTS  = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────
MAX_LEN    = 128      # shorter for transformers (speed)
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR         = 2e-5
WARMUP     = 0.1      # fraction of steps
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────
class TransformerDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Training ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out    = model(**batch)
        loss   = out.loss
        logits = out.logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item() * batch["labels"].size(0)
        correct    += (logits.argmax(1) == batch["labels"]).sum().item()
        total      += batch["labels"].size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for batch in loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out    = model(**batch)
        probs  = torch.softmax(out.logits, dim=1)[:, 1]
        total_loss  += out.loss.item() * batch["labels"].size(0)
        all_preds   += out.logits.argmax(1).cpu().tolist()
        all_labels  += batch["labels"].cpu().tolist()
        all_probs   += probs.cpu().tolist()
    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / n, acc, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="distilbert-base-uncased",
                        help="HuggingFace model hub ID")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    args = parser.parse_args()

    short_name = args.model.replace("/", "_").replace("-", "_")
    print(f"\n{'='*60}")
    print(f" Model : {args.model}")
    print(f" Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Datasets ──────────────────────────────────────────────────────
    print("Tokenizing train set …")
    train_ds = TransformerDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN
    )
    print("Tokenizing test set …")
    test_ds = TransformerDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LEN
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model} …")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Optimizer & scheduler ─────────────────────────────────────────
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────
    history  = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1  = 0.0
    ckpt_path = RESULTS / f"{short_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, test_loader)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    # ── Final evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    t0 = time.perf_counter()
    _, final_acc, final_f1, final_auc = evaluate(model, test_loader)
    infer_ms = (time.perf_counter() - t0) / len(test_ds) * 1000

    print(f"\n✅  Best checkpoint results:")
    print(f"   Accuracy : {final_acc:.4f}")
    print(f"   Macro F1 : {final_f1:.4f}")
    print(f"   ROC-AUC  : {final_auc:.4f}")
    print(f"   Infer (ms/sample): {infer_ms:.3f}")

    metrics = {
        "model": args.model,
        "accuracy": round(final_acc, 4),
        "macro_f1": round(final_f1, 4),
        "roc_auc":  round(final_auc, 4),
        "infer_ms_per_sample": round(infer_ms, 4),
        "epochs": args.epochs,
        "trainable_params": total_params,
    }
    out_path = RESULTS / f"{short_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄  Metrics saved → {out_path}")

    # ── Training curve ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ep = range(1, args.epochs + 1)

    ax1.plot(ep, history["train_loss"], label="Train Loss")
    ax1.plot(ep, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{args.model} — Loss")
    ax1.legend()

    ax2.plot(ep, history["val_acc"], label="Accuracy")
    ax2.plot(ep, history["val_f1"],  label="Macro F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title(f"{args.model} — Metrics")
    ax2.legend()

    plt.tight_layout()
    curve_path = RESULTS / f"{short_name}_training_curve.png"
    plt.savefig(curve_path, dpi=150)
    print(f"📊  Training curve saved → {curve_path}")


if __name__ == "__main__":
    main()
