"""
models/deep_learning/textcnn.py
────────────────────────────────
TextCNN model for binary sentiment classification.
Kim (2014) — "Convolutional Neural Networks for Sentence Classification"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int        = 100,
        num_filters:   int        = 128,
        filter_sizes:  list[int]  = None,
        num_classes:   int        = 2,
        dropout:       float      = 0.5,
        pretrained_embeddings     = None,
        freeze_embeddings:   bool = False,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                         # (batch, seq, embed)
        emb = emb.permute(0, 2, 1)                      # (batch, embed, seq)  ← Conv1d expects this

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                        # (batch, filters, seq-k+1)
            p = F.max_pool1d(c, kernel_size=c.shape[2]) # (batch, filters, 1)
            pooled.append(p.squeeze(2))                  # (batch, filters)

        cat    = torch.cat(pooled, dim=1)                # (batch, filters * len(filter_sizes))
        logits = self.fc(self.dropout(cat))              # (batch, num_classes)
        return logits
