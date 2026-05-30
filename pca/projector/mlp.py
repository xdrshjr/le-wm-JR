"""WorldModelProjector — LLaVA-style 2-layer MLP, fp16 (R01).

R4 (wm-llm-alignment §2.3): the projector now emits ``num_tokens`` soft
tokens per latent instead of a single prefix token, so the aligned model
can prepend K LLM-space tokens carrying the world model's predicted
outcome (spec §2.2). ``num_tokens=1`` reproduces the original single-token
behaviour bit-for-bit, keeping the Two-Room / Stage-1 stub callers valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WorldModelProjectorConfig:
    in_dim: int = 384
    hidden_dim: int = 1024
    out_dim: int = 1536
    num_tokens: int = 1  # K soft tokens per latent (wm-llm-alignment §2.3).
    dtype: torch.dtype = torch.float16  # Turing has no bf16 — see R01.
    zero_init_last: bool = True  # LLaVA-1.5 default (TODO 04.3 / OQ4.5).


class WorldModelProjector(nn.Module):
    """Project world-model latent ``z ∈ R^in_dim`` to ``K`` LLM input
    embeddings ``z_proj ∈ R^{K×out_dim}`` (K = ``cfg.num_tokens``).

    ``forward`` returns ``(B, K, out_dim)``; with ``num_tokens=1`` this is
    the original single-prefix-token shape, so existing callers are
    unaffected (spec §2.3, backward compatible).
    """

    def __init__(self, cfg: WorldModelProjectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.num_tokens * cfg.out_dim)
        self.norm = nn.LayerNorm(cfg.out_dim)

        if cfg.zero_init_last:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        self.to(dtype=cfg.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dtype != self.cfg.dtype:
            z = z.to(self.cfg.dtype)
        x = self.fc1(z)
        x = self.act(x)
        x = self.fc2(x)  # (B, K * out_dim)
        x = x.view(x.size(0), self.cfg.num_tokens, self.cfg.out_dim)
        x = self.norm(x)
        return x  # (B, K, out_dim)
