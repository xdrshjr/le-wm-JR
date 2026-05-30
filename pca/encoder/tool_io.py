"""ToolIOEncoder — frozen LM base + transformer head → R^out_dim.

Spec §2.2 + §4.1 + R06/R07. Single ``cfg`` parameter keeps ctor at
2 args incl. self (under MAX_FUNCTION_PARAMS=5).

Two encoder shapes, selected by ``cfg`` (backward compatible — the MVP /
goal_dist configs keep the original CLS + length-1-head path bit-for-bit):

  legacy (``pooling='cls'``, ``head_over_tokens=False``):
      base(frozen) → [CLS] hidden → Linear → head(seq-len-1) → Linear
  token-head (``pooling='mean'``, ``head_over_tokens=True``):
      base(frozen) → per-token hidden → Linear → head over the *full*
      token sequence (padding-masked) → masked-mean pool → Linear

The token-head path is what the HumanEval verifier uses: it lets the
trainable head attend over code tokens (the discriminative signal) and
mean-pools the way sentence-/code-LM bases are actually trained, instead
of reading a single near-meaningless CLS vector (Round-3 fix).
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
    # Round-3 (backward-compatible) knobs. Defaults reproduce the legacy
    # CLS + seq-len-1-head behaviour, so MVP / goal_dist configs are
    # unchanged; the HumanEval verifier config opts into mean + token-head.
    pooling: str = "cls"            # "cls" | "mean"
    head_over_tokens: bool = False  # run head over full token sequence


class ToolIOEncoder(nn.Module):
    """Encode a batch of free-form tool-IO text strings into R^out_dim."""

    def __init__(self, cfg: ToolIOEncoderConfig) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def _base_hidden(self, enc: dict) -> torch.Tensor:
        if self.cfg.freeze_base:
            with torch.no_grad():
                out = self.base(**enc)
        else:
            out = self.base(**enc)
        return out.last_hidden_state  # (B, L, base_hidden)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over valid tokens. x:(B,L,D) mask:(B,L) -> (B,D)."""
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros(0, self.cfg.out_dim, device=self._device)

        enc = self._tokenize(texts)
        hidden = self._base_hidden(enc)  # (B, L, base_hidden)
        mask = enc["attention_mask"]      # (B, L)

        if self.cfg.head_over_tokens:
            x = self.input_proj(hidden)              # (B, L, hidden)
            pad = mask == 0                          # (B, L) True = pad
            x = self.head(x, src_key_padding_mask=pad)
            pooled = self._masked_mean(x, mask)      # (B, hidden)
        else:
            if self.cfg.pooling == "mean":
                base_vec = self._masked_mean(hidden, mask)  # (B, base_hidden)
            else:  # legacy CLS
                base_vec = hidden[:, 0]
            x = self.input_proj(base_vec).unsqueeze(1)  # (B, 1, hidden)
            x = self.head(x)
            pooled = x[:, 0]
        return self.output_proj(pooled)  # (B, out_dim)

    def state_dict(self, *args, **kwargs):
        """Drop the frozen base LM from checkpoints (Round-3 fix).

        The base is reloaded from HF at construction, so persisting its
        weights every epoch is pure waste — a Qwen-0.5B base adds ~2 GB
        per checkpoint and filled the disk. Trainable head / proj weights
        are still saved; ``load_state_dict(..., strict=False)`` then leaves
        the base at its pretrained init (correct, since it is frozen).
        """
        sd = super().state_dict(*args, **kwargs)
        if not self.cfg.freeze_base:
            return sd
        prefix = kwargs.get("prefix", "")
        if not prefix and len(args) >= 2 and isinstance(args[1], str):
            prefix = args[1]
        drop = f"{prefix}base."
        for k in [k for k in sd if k.startswith(drop)]:
            del sd[k]
        return sd

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device
