"""
models/deep_learning/cnn_model.py
───────────────────────────────────
TextCNN sentiment classifier (Kim 2014).

Architecture:
    Embedding (vocab_size, embed_dim)           [trainable, optional GloVe]
        -> Dropout
        -> Parallel Conv1d layers (filter_sizes = [2, 3, 4])
            Each: Conv1d(embed_dim, num_filters, k) -> ReLU -> AdaptiveMaxPool1d(1)
        -> Concatenate: (batch, num_filters * len(filter_sizes))
        -> Dropout
        -> Linear(num_filters * len(filter_sizes), num_classes)

Why multiple kernel sizes?
    - kernel_size=2: bigram patterns ("not good", "very bad")
    - kernel_size=3: trigram patterns ("not at all", "works as expected")
    - kernel_size=4: 4-gram patterns (longer sentiment phrases)
    Combining them lets the model detect sentiment at multiple granularities
    simultaneously without any recurrence — enabling fast parallel inference.

Key difference from LSTM:
    CNN has no sequential dependency. Each filter reads the entire sequence in
    one pass. This makes it significantly faster than LSTM at inference time
    but limits its ability to capture long-range dependencies.
"""

import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from models.deep_learning.config import (
    EMBED_DIM, MAX_LEN, PAD_IDX, NUM_CLASSES,
    CNN_CONFIG, CNN_MODEL_CONFIG_FILE, GLOVE_PATH,
)


class TextCNN(nn.Module):
    """
    Multi-filter TextCNN for binary sentiment classification.

    Args:
        vocab_size:    Vocabulary size.
        embed_dim:     Embedding dimensionality.
        num_filters:   Number of filters per kernel size.
        filter_sizes:  List of kernel widths (e.g. [2, 3, 4]).
        num_classes:   Output classes (2 for binary).
        dropout:       Dropout probability.
        pad_idx:       Padding token index.
        pretrained_embeddings: Optional (vocab_size, embed_dim) numpy array.
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int        = EMBED_DIM,
        num_filters:  int        = CNN_CONFIG["num_filters"],
        filter_sizes: list[int]  = CNN_CONFIG["filter_sizes"],
        num_classes:  int        = NUM_CLASSES,
        dropout:      float      = CNN_CONFIG["dropout"],
        pad_idx:      int        = PAD_IDX,
        pretrained_embeddings: np.ndarray | None = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )

        # One Conv1d per filter size — all run in parallel
        # Input to Conv1d is (batch, in_channels=embed_dim, seq_len)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels  = embed_dim,
                out_channels = num_filters,
                kernel_size  = k,
            )
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) integer token indices

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed and transpose for Conv1d: (B, L, E) -> (B, E, L)
        emb = self.dropout(self.embedding(x)).permute(0, 2, 1)   # (B, E, L)

        # Apply each conv filter, ReLU, then global max-pool to scalar per filter
        pooled = []
        for conv in self.convolutions:
            c = torch.relu(conv(emb))           # (B, num_filters, L - k + 1)
            # AdaptiveMaxPool1d(1) → (B, num_filters, 1) → squeeze → (B, num_filters)
            c = torch.nn.functional.adaptive_max_pool1d(c, 1).squeeze(2)
            pooled.append(c)

        # Concatenate across all filter sizes: (B, num_filters * len(filter_sizes))
        combined = torch.cat(pooled, dim=1)
        logits   = self.fc(self.dropout(combined))
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model config serialization ────────────────────────────────────────────────

def save_cnn_config(vocab_size: int, **overrides) -> pathlib.Path:
    """Save TextCNN hyperparameters to JSON for reproducible loading."""
    config = {
        "vocab_size":   vocab_size,
        "embed_dim":    overrides.get("embed_dim",    EMBED_DIM),
        "num_filters":  overrides.get("num_filters",  CNN_CONFIG["num_filters"]),
        "filter_sizes": overrides.get("filter_sizes", CNN_CONFIG["filter_sizes"]),
        "num_classes":  overrides.get("num_classes",  NUM_CLASSES),
        "dropout":      overrides.get("dropout",      CNN_CONFIG["dropout"]),
        "pad_idx":      PAD_IDX,
        "max_len":      MAX_LEN,
    }
    CNN_MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CNN_MODEL_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return CNN_MODEL_CONFIG_FILE


def load_cnn_config() -> dict:
    with open(CNN_MODEL_CONFIG_FILE) as f:
        return json.load(f)
