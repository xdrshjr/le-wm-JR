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


class Trajectory(BaseModel):
    instance_id: str
    source: Literal["swegym", "self_replay", "teacher_distill"]
    steps: list[Step]
