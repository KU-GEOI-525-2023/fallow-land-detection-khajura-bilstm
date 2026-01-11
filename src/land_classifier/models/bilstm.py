from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


class BiLSTM(nn.Module):
    """Bidirectional LSTM for land cover classification from time-series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout_temporal = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.training:
            sequence = self.dropout_temporal(sequence)

        _, (hidden, _) = self.encoder(sequence)
        forward_state, backward_state = hidden[-2], hidden[-1]
        combined = torch.cat([forward_state, backward_state], dim=-1)
        return self.classifier(self.ln(combined))


def build_model(cfg: DictConfig | dict) -> BiLSTM:
    """Instantiate BiLSTM from config."""
    cfg = OmegaConf.create(cfg)
    return BiLSTM(
        input_dim=cfg.get("input_dim", 3),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.0),
    )
