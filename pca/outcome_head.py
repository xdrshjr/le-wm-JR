"""OutcomeHead — predicts P(visible tests pass) from a predicted latent.

A small 2-layer MLP mapping the world model's predicted next-state
embedding ẑ₁ ∈ R^embed_dim to a single pass/fail logit. Trained with
``BCEWithLogits`` against the visible-assert pass ratio (spec §4.2,
§5.1), and used by ``WMReranker`` in verifier mode to rank candidates
without executing them.
"""
from __future__ import annotations

from torch import nn


class OutcomeHead(nn.Module):
    """2-layer MLP: latent embedding → scalar pass logit."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        """z: (..., in_dim) → logit (..., 1). Callers squeeze the last dim."""
        return self.net(z)
