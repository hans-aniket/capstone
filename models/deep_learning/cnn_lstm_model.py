"""
models/deep_learning/cnn_lstm_model.py
────────────────────────────────────────
Hybrid CNN-LSTM sentiment classifier.

Architecture:
    Embedding (vocab_size, embed_dim)       [trainable, optional GloVe]
        -> Dropout
        -> Conv1d(embed_dim, num_filters, kernel_size)
            — captures local n-gram features at every sequence position
        -> ReLU
        -> BiLSTM(num_filters, lstm_hidden, bidirectional=True)
            — models the sequence of those local features
        -> Concatenate last fwd + bwd hidden states
        -> Dropout
        -> Linear(lstm_hidden * 2, num_classes)

Why this hybrid?
    - Pure CNN (TextCNN): globally pools features → loses positional information
    - Pure LSTM: sees raw words → must infer n-gram patterns implicitly
    - CNN-LSTM: CNN detects local patterns explicitly, LSTM models their
      sequential arrangement → combines strengths of both

The CNN acts as a learned feature extractor (like n-gram detectors), and the
LSTM reads the sequence of those features with full temporal context.

Key design choice: single kernel size (trigrams by default) to keep the
sequence length uniform for LSTM. Multiple kernel sizes would require
padding to a common length before feeding the LSTM — an unnecessary complication
for marginal benefit given the LSTM's capacity.
"""

import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from models.deep_learning.config import (
    EMBED_DIM, MAX_LEN, PAD_IDX, NUM_CLASSES,
    CNN_LSTM_CONFIG, CNN_LSTM_MODEL_CONFIG_FILE,
)


class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN-LSTM for binary sentiment classification.

    Args:
        vocab_size:    Vocabulary size.
        embed_dim:     Embedding dimensionality.
        num_filters:   Conv1d output channels.
        kernel_size:   Convolutional kernel width.
        lstm_hidden:   LSTM hidden state size (per direction).
        lstm_layers:   Number of stacked LSTM layers.
        num_classes:   Output classes.
        bidirectional: If True, use BiLSTM.
        dropout:       Dropout probability.
        pad_idx:       Padding token index.
        pretrained_embeddings: Optional numpy array (vocab_size, embed_dim).
    """

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = EMBED_DIM,
        num_filters:   int   = CNN_LSTM_CONFIG["num_filters"],
        kernel_size:   int   = CNN_LSTM_CONFIG["kernel_size"],
        lstm_hidden:   int   = CNN_LSTM_CONFIG["lstm_hidden"],
        lstm_layers:   int   = CNN_LSTM_CONFIG["lstm_layers"],
        num_classes:   int   = NUM_CLASSES,
        bidirectional: bool  = CNN_LSTM_CONFIG["bidirectional"],
        dropout:       float = CNN_LSTM_CONFIG["dropout"],
        pad_idx:       int   = PAD_IDX,
        pretrained_embeddings: np.ndarray | None = None,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.lstm_hidden   = lstm_hidden

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )

        # ── Local feature extractor (CNN) ─────────────────────────────────────
        # Input:  (B, E, L)        [after permute]
        # Output: (B, num_filters, L - kernel_size + 1)
        self.conv = nn.Conv1d(
            in_channels  = embed_dim,
            out_channels = num_filters,
            kernel_size  = kernel_size,
        )

        # ── Sequential modeler (BiLSTM) ───────────────────────────────────────
        # Input:  (B, L - kernel_size + 1, num_filters)  [after permute]
        self.lstm = nn.LSTM(
            input_size   = num_filters,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.dropout  = nn.Dropout(dropout)
        lstm_out_dim  = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc       = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) integer token indices

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed + dropout: (B, L, E)
        emb = self.dropout(self.embedding(x))

        # CNN expects (B, E, L)
        emb_t = emb.permute(0, 2, 1)                           # (B, E, L)
        conv_out = torch.relu(self.conv(emb_t))                 # (B, F, L')
        # L' = L - kernel_size + 1

        # LSTM expects (B, L', F)
        lstm_in     = conv_out.permute(0, 2, 1)                 # (B, L', F)
        _, (hn, _)  = self.lstm(lstm_in)
        # hn: (num_layers * num_directions, B, lstm_hidden)

        if self.bidirectional:
            combined = torch.cat([hn[-2], hn[-1]], dim=1)       # (B, lstm_hidden * 2)
        else:
            combined = hn[-1]                                   # (B, lstm_hidden)

        logits = self.fc(self.dropout(combined))                # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model config serialization ────────────────────────────────────────────────

def save_cnn_lstm_config(vocab_size: int, **overrides) -> pathlib.Path:
    """Save CNN-LSTM hyperparameters to JSON for reproducible loading."""
    config = {
        "vocab_size":    vocab_size,
        "embed_dim":     overrides.get("embed_dim",     EMBED_DIM),
        "num_filters":   overrides.get("num_filters",   CNN_LSTM_CONFIG["num_filters"]),
        "kernel_size":   overrides.get("kernel_size",   CNN_LSTM_CONFIG["kernel_size"]),
        "lstm_hidden":   overrides.get("lstm_hidden",   CNN_LSTM_CONFIG["lstm_hidden"]),
        "lstm_layers":   overrides.get("lstm_layers",   CNN_LSTM_CONFIG["lstm_layers"]),
        "num_classes":   overrides.get("num_classes",   NUM_CLASSES),
        "bidirectional": overrides.get("bidirectional", CNN_LSTM_CONFIG["bidirectional"]),
        "dropout":       overrides.get("dropout",       CNN_LSTM_CONFIG["dropout"]),
        "pad_idx":       PAD_IDX,
        "max_len":       MAX_LEN,
    }
    CNN_LSTM_MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CNN_LSTM_MODEL_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return CNN_LSTM_MODEL_CONFIG_FILE


def load_cnn_lstm_config() -> dict:
    with open(CNN_LSTM_MODEL_CONFIG_FILE) as f:
        return json.load(f)
