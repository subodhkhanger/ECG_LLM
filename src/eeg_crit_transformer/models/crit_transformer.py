from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L]
        return self.dropout(x)


@dataclass
class CritTransformerConfig:
    in_channels: int
    patch_size: int = 64
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_classes: int = 2


class CritTransformer(nn.Module):
    """
    Minimal multichannel 1D time-series Transformer.

    - Splits time axis into non-overlapping patches of length `patch_size`.
    - Each patch is flattened across channels and linearly projected to `d_model`.
    - Transformer encoder processes the patch sequence with positional encoding.
    - CLS token pooling for classification.
    """

    def __init__(self, cfg: CritTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = nn.Linear(cfg.in_channels * cfg.patch_size, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_encoding = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        B, C, T = x.shape
        assert C == self.cfg.in_channels, f"Expected {self.cfg.in_channels} channels, got {C}"
        assert T % self.cfg.patch_size == 0, "Time length must be multiple of patch_size"

        # (B, C, T) -> (B, num_patches, C*patch)
        num_patches = T // self.cfg.patch_size
        x = x.unfold(dimension=2, size=self.cfg.patch_size, step=self.cfg.patch_size)
        # x: (B, C, num_patches, patch)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, num_patches, C * self.cfg.patch_size)
        x = self.patch_embed(x)  # (B, L, D)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, D)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS
        logits = self.head(x)
        return logits


def build_model(
    in_channels: int,
    num_classes: int = 2,
    patch_size: int = 64,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> CritTransformer:
    cfg = CritTransformerConfig(
        in_channels=in_channels,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes,
    )
    return CritTransformer(cfg)

