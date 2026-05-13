"""
models/deep_learning/trainer.py
────────────────────────────────
LSTMTrainer: training loop, validation loop, early stopping,
checkpointing, and training history for the LSTM sentiment classifier.

Implements BaseTrainer for integration with the comparative system.
"""

import json
import time
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt

from models.base import BaseTrainer
from models.deep_learning.config import (
    DEVICE, TRAIN_CONFIG,
    LSTM_CHECKPOINT_BEST, LSTM_CHECKPOINT_LAST,
    LSTM_TRAINING_HISTORY, LSTM_RESULTS_DIR,
)

logger = logging.getLogger(__name__)


class LSTMTrainer(BaseTrainer):
    """
    Training orchestrator for LSTMClassifier.

    Usage:
        trainer = LSTMTrainer(model)
        history = trainer.train(train_loader, val_loader)
        trainer.load_best_checkpoint()
        y_pred  = trainer.predict(test_loader)
        y_prob  = trainer.predict_proba(test_loader)
    """

    name: str = "LSTM"
    tier: str = "deep_learning"

    def __init__(
        self,
        model:       nn.Module,
        device:      torch.device    = DEVICE,
        lr:          float           = TRAIN_CONFIG["lr"],
        weight_decay:float           = TRAIN_CONFIG["weight_decay"],
        clip_norm:   float           = TRAIN_CONFIG["clip_grad_norm"],
        patience:    int             = TRAIN_CONFIG["patience"],
    ):
        self.model       = model.to(device)
        self.device      = device
        self.clip_norm   = clip_norm
        self.patience    = patience
        self._trained    = False

        self.optimizer  = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=1, factor=0.5
        )
        self.criterion  = nn.CrossEntropyLoss()

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int = TRAIN_CONFIG["num_epochs"],
    ) -> dict:
        """
        Train with early stopping and checkpoint saving.

        Returns:
            history dict with per-epoch train/val loss and metrics.
        """
        logger.info("[LSTM] Starting training | device=%s  epochs=%d  patience=%d",
                    self.device, num_epochs, self.patience)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }
        best_val_f1    = 0.0
        no_improve     = 0
        total_train_time = 0.0

        for epoch in range(1, num_epochs + 1):
            t0 = time.perf_counter()

            tr_loss, tr_acc             = self._train_epoch(train_loader)
            val_loss, val_acc, val_f1   = self._validate_epoch(val_loader)
            self.scheduler.step(val_f1)

            epoch_time = time.perf_counter() - t0
            total_train_time += epoch_time

            history["train_loss"].append(round(tr_loss,  4))
            history["train_acc"].append(round(tr_acc,    4))
            history["val_loss"].append(round(val_loss,   4))
            history["val_acc"].append(round(val_acc,     4))
            history["val_f1"].append(round(val_f1,       4))

            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
                no_improve  = 0
                self._save_checkpoint(LSTM_CHECKPOINT_BEST, epoch, val_f1)
                tag = "  [NEW BEST]"
            else:
                no_improve += 1
                tag = ""

            logger.info(
                "Epoch %2d/%d | train_loss=%.4f acc=%.4f | "
                "val_loss=%.4f acc=%.4f f1=%.4f | %.1fs%s",
                epoch, num_epochs, tr_loss, tr_acc,
                val_loss, val_acc, val_f1, epoch_time, tag,
            )

            self._save_checkpoint(LSTM_CHECKPOINT_LAST, epoch, val_f1)

            if no_improve >= self.patience:
                logger.info("[LSTM] Early stopping triggered after %d epochs with no improvement.", epoch)
                break

        history["best_val_f1"]     = round(best_val_f1, 4)
        history["total_train_time_s"] = round(total_train_time, 2)
        self._trained = True

        self._save_history(history)
        self._plot_curves(history)
        return history

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        for x, y in loader:
            x, y   = x.to(self.device), y.to(self.device)
            logits  = self.model(x)
            loss    = self.criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            all_preds  += logits.argmax(1).cpu().tolist()
            all_labels += y.cpu().tolist()

        n   = len(all_labels)
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / n, acc, f1

    # ── BaseTrainer interface ─────────────────────────────────────────────────

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Run inference on a DataLoader; return integer predictions."""
        self._assert_trained()
        return self._run_inference(loader)[0]

    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """
        Run inference on a DataLoader; return (N, 2) probability array.
        Column 1 = P(positive).
        """
        self._assert_trained()
        _, proba = self._run_inference(loader)
        return proba

    @torch.no_grad()
    def _run_inference(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Returns (y_pred, proba) arrays."""
        self.model.eval()
        all_preds, all_proba = [], []

        for x, *_ in loader:   # labels may or may not be present
            x      = x.to(self.device)
            logits = self.model(x)
            proba  = torch.softmax(logits, dim=1)
            all_preds  += logits.argmax(1).cpu().tolist()
            all_proba.append(proba.cpu().numpy())

        return (
            np.array(all_preds, dtype=np.int32),
            np.vstack(all_proba),
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        path:    pathlib.Path,
        epoch:   int,
        val_f1:  float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch":               epoch,
                "val_f1":              val_f1,
                "model_state_dict":    self.model.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict(),
            },
            path,
        )

    def load_best_checkpoint(self) -> dict:
        """Load the best checkpoint back into the model (call before test evaluation)."""
        if not LSTM_CHECKPOINT_BEST.exists():
            raise FileNotFoundError(f"No checkpoint found at {LSTM_CHECKPOINT_BEST}")
        ckpt = torch.load(LSTM_CHECKPOINT_BEST, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("[LSTM] Best checkpoint loaded (epoch=%d  val_f1=%.4f)",
                    ckpt["epoch"], ckpt["val_f1"])
        self._trained = True
        return ckpt

    def save(self, directory: pathlib.Path | None = None) -> pathlib.Path:
        """Alias: save the best checkpoint (satisfies BaseTrainer interface)."""
        return LSTM_CHECKPOINT_BEST

    @classmethod
    def load(cls, directory: pathlib.Path | None = None) -> "LSTMTrainer":
        """Not used directly — use LSTMPredictor.from_saved() for inference."""
        raise NotImplementedError("Use LSTMPredictor.from_saved() for inference loading.")

    # ── History and plots ─────────────────────────────────────────────────────

    def _save_history(self, history: dict) -> None:
        LSTM_TRAINING_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with open(LSTM_TRAINING_HISTORY, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("[LSTM] Training history saved -> %s", LSTM_TRAINING_HISTORY)

    def _plot_curves(self, history: dict) -> None:
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("LSTM Training Curves — amazon_polarity", weight="bold")

        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
        ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, history["val_acc"], label="Val Accuracy", marker="o", markersize=3)
        ax2.plot(epochs, history["val_f1"],  label="Val F1 Macro", marker="s", markersize=3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
        ax2.set_title("Metrics"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = LSTM_RESULTS_DIR / "training_curves.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("[LSTM] Training curves saved -> %s", out)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "LSTMTrainer: model is not trained. "
                "Call train() or load_best_checkpoint() first."
            )
