"""
models/deep_learning/bilstm.py
───────────────────────────────
Bidirectional LSTM model for binary sentiment classification.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 100,
        hidden_dim:    int   = 128,
        num_layers:    int   = 2,
        num_classes:   int   = 2,
        dropout:       float = 0.3,
        pretrained_embeddings=None,
        freeze_embeddings:    bool  = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)   # *2 for bidirectional

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))                    # (batch, seq, embed)
        out, (hn, _) = self.lstm(emb)                            # hn: (layers*2, batch, hidden)
        # Concatenate last forward & backward hidden states
        fwd = hn[-2]   # last forward layer
        bwd = hn[-1]   # last backward layer
        combined = torch.cat([fwd, bwd], dim=1)                  # (batch, hidden*2)
        logits = self.fc(self.dropout(combined))                 # (batch, num_classes)
        return logits
