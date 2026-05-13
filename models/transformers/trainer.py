"""
models/transformers/trainer.py
───────────────────────────────
TransformerTrainer: Fine-tuning loop for HuggingFace sequence classification models.
Supports mixed precision (AMP) for faster training on CUDA.
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
from transformers import get_linear_schedule_with_warmup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.base import BaseTrainer
from models.transformers.config import DEVICE, TRAIN_CONFIG

logger = logging.getLogger(__name__)


class TransformerTrainer(BaseTrainer):
    """
    Fine-tunes a HuggingFace AutoModelForSequenceClassification model.
    Includes mixed precision (AMP) if CUDA is available.
    """

    tier: str = "transformers"

    def __init__(
        self,
        model:        nn.Module,
        model_name:   str,
        ckpt_best:    pathlib.Path,
        ckpt_last:    pathlib.Path,
        history_path: pathlib.Path,
        results_dir:  pathlib.Path,
        device:       torch.device = DEVICE,
        lr:           float        = TRAIN_CONFIG["lr"],
        weight_decay: float        = TRAIN_CONFIG["weight_decay"],
        clip_norm:    float        = TRAIN_CONFIG["clip_grad_norm"],
        patience:     int          = TRAIN_CONFIG["patience"],
    ):
        self.model        = model.to(device)
        self.name         = model_name
        self.ckpt_best    = ckpt_best
        self.ckpt_last    = ckpt_last
        self.history_path = history_path
        self.results_dir  = results_dir
        self.device       = device
        self.clip_norm    = clip_norm
        self.patience     = patience
        self._trained     = False

        # Apply weight decay to all parameters EXCEPT bias and LayerNorm weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # We will initialize the scheduler in train() once we know the number of steps
        self.scheduler = None

        # Setup AMP (Automatic Mixed Precision)
        self.scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int = TRAIN_CONFIG["num_epochs"],
    ) -> dict:
        """
        Fine-tune the model.
        """
        logger.info(
            "[%s] Training | device=%s  epochs=%d  patience=%d  mixed_precision=%s",
            self.name, self.device, num_epochs, self.patience, bool(self.scaler)
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = TRAIN_CONFIG.get("warmup_steps", 0)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }
        best_val_f1 = 0.0
        no_improve  = 0
        total_time  = 0.0

        for epoch in range(1, num_epochs + 1):
            t0 = time.perf_counter()

            tr_loss, tr_acc           = self._train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self._validate_epoch(val_loader)

            elapsed = time.perf_counter() - t0
            total_time += elapsed

            history["train_loss"].append(round(tr_loss,  4))
            history["train_acc"].append(round(tr_acc,    4))
            history["val_loss"].append(round(val_loss,   4))
            history["val_acc"].append(round(val_acc,     4))
            history["val_f1"].append(round(val_f1,       4))

            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
                no_improve  = 0
                self._save_checkpoint(self.ckpt_best, epoch, val_f1)
                tag = "  [NEW BEST]"
            else:
                no_improve += 1
                tag = f"  (no improvement {no_improve}/{self.patience})"

            self._save_checkpoint(self.ckpt_last, epoch, val_f1)

            logger.info(
                "Epoch %2d/%d | train_loss=%.4f acc=%.4f | "
                "val_loss=%.4f acc=%.4f f1=%.4f | %.1fs%s",
                epoch, num_epochs, tr_loss, tr_acc,
                val_loss, val_acc, val_f1, elapsed, tag,
            )

            if no_improve >= self.patience:
                logger.info(
                    "[%s] Early stopping after %d epochs.", self.name, epoch
                )
                break

        history["best_val_f1"]        = round(best_val_f1, 4)
        history["total_train_time_s"] = round(total_time, 2)
        self._trained = True

        self._save_history(history)
        self._plot_curves(history)
        return history

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # If model supports token_type_ids (e.g. BERT but not DistilBERT)
            kwargs = {}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()
                
            self.scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            
        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            kwargs = {}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            all_preds  += logits.argmax(1).cpu().tolist()
            all_labels += labels.cpu().tolist()
            
        n   = len(all_labels)
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / n, acc, f1

    # ── BaseTrainer interface ─────────────────────────────────────────────────

    def predict(self, loader: DataLoader) -> np.ndarray:
        self._assert_trained()
        return self._run_inference(loader)[0]

    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        self._assert_trained()
        return self._run_inference(loader)[1]

    @torch.no_grad()
    def _run_inference(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_proba = [], []
        
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            kwargs = {}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            logits = outputs.logits
            proba  = torch.softmax(logits, dim=1)
            
            all_preds  += logits.argmax(1).cpu().tolist()
            all_proba.append(proba.cpu().numpy())
            
        return (
            np.array(all_preds, dtype=np.int32),
            np.vstack(all_proba),
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, path: pathlib.Path, epoch: int, val_f1: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid weights_only issues with scalars
        torch.save(
            {
                "epoch":                epoch,
                "val_f1":               float(val_f1),
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )

    def load_best_checkpoint(self) -> dict:
        if not self.ckpt_best.exists():
            raise FileNotFoundError(f"No checkpoint at {self.ckpt_best}")
        ckpt = torch.load(self.ckpt_best, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._trained = True
        logger.info(
            "[%s] Best checkpoint loaded (epoch=%d  val_f1=%.4f)",
            self.name, ckpt["epoch"], ckpt["val_f1"],
        )
        return ckpt

    def save(self, directory: pathlib.Path | None = None) -> pathlib.Path:
        return self.ckpt_best

    @classmethod
    def load(cls, directory: pathlib.Path | None = None) -> "TransformerTrainer":
        raise NotImplementedError("Use TransformerPredictor.from_saved() for inference.")

    # ── History + curves ──────────────────────────────────────────────────────

    def _save_history(self, history: dict) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def _plot_curves(self, history: dict) -> None:
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{self.name} Training Curves — amazon_polarity", weight="bold")

        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
        ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
        ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, history["val_acc"], label="Val Accuracy", marker="o", markersize=3)
        ax2.plot(epochs, history["val_f1"],  label="Val F1 Macro", marker="s", markersize=3)
        ax2.set(xlabel="Epoch", ylabel="Score", title="Metrics")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.results_dir / "training_curves.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("[%s] Training curves saved -> %s", self.name, out)

    def _assert_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                f"[{self.name}] Not trained. Call train() or load_best_checkpoint() first."
            )
