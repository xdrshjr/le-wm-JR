"""WorldModelProjector — LLaVA-style 2-layer MLP, fp16 (R01)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WorldModelProjectorConfig:
    in_dim: int = 384
    hidden_dim: int = 1024
    out_dim: int = 1536
    dtype: torch.dtype = torch.float16  # Turing has no bf16 — see R01.
    zero_init_last: bool = True  # LLaVA-1.5 default (TODO 04.3 / OQ4.5).


class WorldModelProjector(nn.Module):
    """Project world-model latent ``z ∈ R^in_dim`` to LLM input embedding
    ``z_proj ∈ R^{1×out_dim}`` (one prefix token, MVP — OQ4.1 deferred to v2).
    """

    def __init__(self, cfg: WorldModelProjectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)
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
        x = self.fc2(x)
        x = self.norm(x)
        return x.unsqueeze(1)  # (B, 1, out_dim)
