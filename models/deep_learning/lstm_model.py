"""
models/deep_learning/lstm_model.py
────────────────────────────────────
Bidirectional LSTM sentiment classifier.

Architecture:
    Embedding (trainable, optional GloVe init)
        -> Dropout
        -> BiLSTM (num_layers stacked)
        -> Concatenate final forward + backward hidden states
        -> Dropout
        -> Linear(hidden_dim * 2, num_classes)

Why bidirectional?
    Reviews contain sentiment signals that depend on context from both
    directions ("not bad at all" requires left AND right context to interpret
    each word). BiLSTM captures both.

Why concatenate final hidden states (not mean-pool outputs)?
    The final hidden states encode the model's summary representation after
    reading the entire sequence. This is the standard approach for sentence-
    level classification with LSTMs.
"""

import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from models.deep_learning.config import (
    EMBED_DIM, MAX_LEN, PAD_IDX, NUM_CLASSES,
    LSTM_CONFIG, LSTM_MODEL_CONFIG_FILE, GLOVE_PATH,
)


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Args:
        vocab_size:   Total vocabulary size (from Vocabulary object).
        embed_dim:    Embedding dimensionality.
        hidden_dim:   LSTM hidden state size (per direction).
        num_layers:   Number of stacked LSTM layers.
        num_classes:  Output classes (2 for binary).
        dropout:      Dropout probability (applied to embeddings and LSTM output).
        bidirectional: If True, use BiLSTM (recommended).
        pad_idx:      Padding token index — embedding row is zeroed.
        pretrained_embeddings: Optional numpy array of shape (vocab_size, embed_dim).
    """

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = EMBED_DIM,
        hidden_dim:    int   = LSTM_CONFIG["hidden_dim"],
        num_layers:    int   = LSTM_CONFIG["num_layers"],
        num_classes:   int   = NUM_CLASSES,
        dropout:       float = LSTM_CONFIG["dropout"],
        bidirectional: bool  = LSTM_CONFIG["bidirectional"],
        pad_idx:       int   = PAD_IDX,
        pretrained_embeddings: np.ndarray | None = None,
    ):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )
            # Keep embedding trainable — allows domain adaptation during fine-tuning

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc      = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) integer token indices

        Returns:
            logits: (batch_size, num_classes) — raw scores (pre-softmax)
        """
        # Embed + dropout
        emb = self.dropout(self.embedding(x))           # (B, L, E)

        # LSTM — hn shape: (num_layers * num_directions, B, hidden_dim)
        _, (hn, _) = self.lstm(emb)

        if self.bidirectional:
            # Last layer: forward direction = hn[-2], backward = hn[-1]
            fwd      = hn[-2]                           # (B, hidden_dim)
            bwd      = hn[-1]                           # (B, hidden_dim)
            combined = torch.cat([fwd, bwd], dim=1)     # (B, hidden_dim * 2)
        else:
            combined = hn[-1]                           # (B, hidden_dim)

        logits = self.fc(self.dropout(combined))        # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── GloVe loader ──────────────────────────────────────────────────────────────

def load_glove(
    vocab_token2idx: dict[str, int],
    embed_dim: int = EMBED_DIM,
    glove_path: pathlib.Path = GLOVE_PATH,
) -> np.ndarray | None:
    """
    Load GloVe embeddings and build a matrix aligned with the vocabulary.

    Returns:
        numpy array of shape (vocab_size, embed_dim), or None if file missing.
    """
    if not glove_path.exists():
        return None

    vocab_size   = len(vocab_token2idx)
    embed_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    embed_matrix[0] = 0.0   # <PAD> is the zero vector

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            token = parts[0]
            if token in vocab_token2idx:
                embed_matrix[vocab_token2idx[token]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    coverage = found / max(vocab_size - 2, 1) * 100   # exclude PAD and UNK
    print(f"[GloVe] Coverage: {found:,}/{vocab_size:,} tokens ({coverage:.1f}%)")
    return embed_matrix


# ── Model config serialization ────────────────────────────────────────────────

def save_model_config(vocab_size: int, **kwargs) -> pathlib.Path:
    """Save model hyperparameters to JSON for reproducible loading."""
    config = {
        "vocab_size":    vocab_size,
        "embed_dim":     kwargs.get("embed_dim",    EMBED_DIM),
        "hidden_dim":    kwargs.get("hidden_dim",   LSTM_CONFIG["hidden_dim"]),
        "num_layers":    kwargs.get("num_layers",   LSTM_CONFIG["num_layers"]),
        "num_classes":   kwargs.get("num_classes",  NUM_CLASSES),
        "dropout":       kwargs.get("dropout",      LSTM_CONFIG["dropout"]),
        "bidirectional": kwargs.get("bidirectional",LSTM_CONFIG["bidirectional"]),
        "pad_idx":       PAD_IDX,
        "max_len":       MAX_LEN,
    }
    LSTM_MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LSTM_MODEL_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return LSTM_MODEL_CONFIG_FILE


def load_model_config() -> dict:
    """Load model hyperparameters from the saved JSON."""
    with open(LSTM_MODEL_CONFIG_FILE) as f:
        return json.load(f)
