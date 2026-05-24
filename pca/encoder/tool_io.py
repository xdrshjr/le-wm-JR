"""ToolIOEncoder — frozen CodeBERT-base + 4-layer transformer head → R^384.

Spec §2.2 + §4.1 + R06/R07. Single ``cfg`` parameter keeps ctor at
2 args incl. self (under MAX_FUNCTION_PARAMS=5).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ToolIOEncoderConfig:
    base_model: str = "microsoft/codebert-base"
    hidden_dim: int = 384
    out_dim: int = 384
    num_head_layers: int = 4
    freeze_base: bool = True
    max_length: int = 512


class ToolIOEncoder(nn.Module):
    """Encode a batch of free-form tool-IO text strings into R^out_dim.

    Layout:
        CodeBERT(frozen) → [CLS] hidden (R^768) → Linear → 4×TransformerEncoder
        → Linear → R^out_dim.
    """

    def __init__(self, cfg: ToolIOEncoderConfig) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.base = AutoModel.from_pretrained(cfg.base_model)
        base_hidden = self.base.config.hidden_size

        if cfg.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.base.eval()

        self.input_proj = nn.Linear(base_hidden, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=8,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.head = nn.TransformerEncoder(layer, num_layers=cfg.num_head_layers)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros(0, self.cfg.out_dim, device=self._device)

        enc = self._tokenize(texts)
        if self.cfg.freeze_base:
            with torch.no_grad():
                base_out = self.base(**enc)
        else:
            base_out = self.base(**enc)

        cls = base_out.last_hidden_state[:, 0]  # (B, base_hidden)
        x = self.input_proj(cls).unsqueeze(1)  # (B, 1, hidden)
        x = self.head(x)
        x = self.output_proj(x[:, 0])  # (B, out_dim)
        return x

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device
