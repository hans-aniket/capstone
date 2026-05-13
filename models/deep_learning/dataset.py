"""
models/deep_learning/dataset.py
────────────────────────────────
PyTorch Dataset and DataLoader factory for DL sentiment models.

Accepts integer-encoded sequences (numpy arrays) produced by
preprocessing/dl_preprocessor.py and wraps them in a PyTorch Dataset.

This module is shared by LSTM, CNN-LSTM, and any future DL model.
It has no model-specific logic.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ReviewSequenceDataset(Dataset):
    """
    PyTorch Dataset for padded integer-index sequences.

    Args:
        sequences: int32 numpy array of shape (N, max_len) from DLPreprocessor.
        labels:    list or array of integer labels (0=negative, 1=positive).
    """

    def __init__(self, sequences: np.ndarray, labels: list | np.ndarray):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_dataloaders(
    X_train:    np.ndarray,
    y_train:    list | np.ndarray,
    X_test:     np.ndarray,
    y_test:     list | np.ndarray,
    batch_size: int   = 64,
    val_split:  float = 0.1,
    seed:       int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders from preprocessed sequences.

    Validation set is carved from the training data using a reproducible split.
    Test set is held completely separate and used only for final evaluation.

    Args:
        X_train / y_train: Training sequences and labels.
        X_test  / y_test:  Test sequences and labels (never used during training).
        batch_size:        Mini-batch size.
        val_split:         Fraction of training data used for validation.
        seed:              Generator seed for reproducible train/val split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_train_ds = ReviewSequenceDataset(X_train, y_train)

    n_val   = int(len(full_train_ds) * val_split)
    n_train = len(full_train_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=generator)

    test_ds = ReviewSequenceDataset(X_test, y_test)

    # num_workers=0 required on Windows (no fork support)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def dataset_info(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
) -> dict:
    """Return a summary dict of dataset sizes for logging."""
    return {
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
        "test_batches":  len(test_loader),
        "train_samples": len(train_loader.dataset),
        "val_samples":   len(val_loader.dataset),
        "test_samples":  len(test_loader.dataset),
    }
