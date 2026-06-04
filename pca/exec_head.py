"""ExecTraceHead — the neural-executor head of the round-8 world model.

Where ``OutcomeHead`` (``pca/outcome_head.py``) collapses *what code computes*
and *what a test expects* into one non-transferable pass/fail scalar, this head
predicts the **output** a candidate produces on a test input, in the frozen
encoder's embedding space, and judges pass/consistency by *comparing predicted
outputs* (spec wm-exec-trace-fusion-sota §2.2):

    predict_output(ẑ₁) -> ô            "what does this code output here"
    embed_output(z_out) -> proj         project an encoded output text
    match_logit(ô, z_out) -> logit      "are these two outputs equal" (scalar)

``σ(match_logit / τ)`` is the probability that two outputs are equal — the
atom from which the PEC matrix is derived (``consensus.exec_pass_from_outputs``)
without ever executing a candidate. ``predict_output`` raises only via the
``TextJEPA`` wrapper; configs without an ``exec_head`` never touch this module
(zero regression, spec §2.2 invariant 1 / §2.2 C-10).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ExecTraceHead(nn.Module):
    """Predict an output embedding ô and score output equality (spec §2.2)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        proj_dim: int = 256,
        init_tau: float = 1.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or in_dim
        self.proj_dim = proj_dim
        # ẑ₁ (predicted latent) → predicted output embedding ô.
        self.predictor = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )
        # Encoded output text latent → the same proj_dim output space.
        self.out_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
        )
        # Bilinear form (manual matmul so it broadcasts over (K, T, P)).
        self.bilinear = nn.Parameter(torch.eye(proj_dim))
        # Learnable temperature for the InfoNCE term (exposed via ``tau``).
        self.log_tau = nn.Parameter(
            torch.tensor(math.log(max(init_tau, 1e-3)))
        )

    @property
    def tau(self) -> torch.Tensor:
        """Positive, lower-bounded contrastive temperature."""
        return self.log_tau.exp().clamp_min(1e-3)

    def predict_output(self, z: torch.Tensor) -> torch.Tensor:
        """Latent ẑ₁ (..., in_dim) → predicted output emb (..., proj_dim)."""
        return self.predictor(z)

    def embed_output(self, z_out: torch.Tensor) -> torch.Tensor:
        """Encoded output latent (..., in_dim) → output emb (..., proj_dim)."""
        return self.out_proj(z_out)

    def match_logit(
        self, o_hat: torch.Tensor, z_out: torch.Tensor
    ) -> torch.Tensor:
        """Output-equality logit for aligned (..., proj_dim) tensors.

        Both arguments live in proj_dim (``predict_output`` / ``embed_output``
        outputs). Returns cosine + bilinear similarity with the trailing dim
        reduced — callers apply ``σ(logit / τ)`` to get P(outputs equal).
        """
        a = F.normalize(o_hat, dim=-1)
        b = F.normalize(z_out, dim=-1)
        cos = (a * b).sum(-1)
        bil = ((a @ self.bilinear) * b).sum(-1)
        return cos + bil
