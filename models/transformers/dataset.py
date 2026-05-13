"""
models/transformers/dataset.py
───────────────────────────────
PyTorch DataLoader factory for Transformer models.
Accepts SentimentDataset objects directly from TransformerPreprocessor.
"""

import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.transformer_preprocessor import SentimentDataset


def build_dataloaders(
    train_ds:   SentimentDataset,
    test_ds:    SentimentDataset,
    batch_size: int   = 16,
    val_split:  float = 0.1,
    seed:       int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders from preprocessed datasets.

    Validation set is carved from the training data using a reproducible split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    n_val   = int(len(train_ds) * val_split)
    n_train = len(train_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_sub_ds, val_sub_ds = random_split(train_ds, [n_train, n_val], generator=generator)

    # Note: num_workers=0 is required on Windows
    train_loader = DataLoader(
        train_sub_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_sub_ds, batch_size=batch_size, shuffle=False,
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
