"""Pydantic trajectory schema (Spec 01 §Output)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from pca.action.schema import ExecutableOp


class Observation(BaseModel):
    """Tool-IO observation: concat of terminal tail + open-file window +
    test-runner output, truncated by ``pca.data.collate``.
    """

    text: str
    cwd: str = "."


class Step(BaseModel):
    obs_text: str
    op: ExecutableOp
    obs_next_text: str | None = None
    expert_intent: str | None = None
    # Pass ratio ∈ [0, 1] of the visible asserts after running ``op``
    # (verifier BCE target; spec §5.1). ``None`` for legacy / unlabeled
    # trajectories — keeps the MVP/synthetic schema backward compatible.
    label: float | None = None


class Trajectory(BaseModel):
    instance_id: str
    source: Literal["swegym", "self_replay", "teacher_distill"]
    steps: list[Step]
