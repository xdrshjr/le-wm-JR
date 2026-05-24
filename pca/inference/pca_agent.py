"""PCA inference loop — STUB (RA8). Bodies land in adaptation-spec v2.

Interface is locked here so downstream eval harnesses (Spec 06/07)
can import ``PCAAgent`` without circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pca.action.schema import ExecutableOp


@dataclass
class PCAAgentConfig:
    k_candidates: int = 8
    rollout_horizon: int = 3
    cem_topk: int = 30


class PCAAgent:
    """Skeleton — `act()` is the public entrypoint; helpers `_propose_intents`,
    `_rollout`, `_select` are factored to keep each method ≤60 lines (v2 fill).
    """

    def __init__(self, cfg: PCAAgentConfig) -> None:
        self.cfg = cfg

    def act(self, task: Any) -> ExecutableOp:
        raise NotImplementedError(
            "PCAAgent.act() — body filled in adaptation-spec v2"
        )

    def _propose_intents(self, k: int) -> list[str]:
        raise NotImplementedError("filled in adaptation-spec v2")

    def _rollout(
        self,
        z_t: torch.Tensor,
        ops: list[ExecutableOp],
        horizon: int,
    ) -> list[torch.Tensor]:
        raise NotImplementedError("filled in adaptation-spec v2")

    def _select(self, rollouts: list[torch.Tensor]) -> int:
        raise NotImplementedError("filled in adaptation-spec v2")
