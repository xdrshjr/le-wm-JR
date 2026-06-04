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
    # Round-8 execution-trace fields (spec wm-exec-trace-fusion-sota §5).
    # ``out_repr`` = ``repr()`` of the candidate's REAL output on the test
    # input (the obs_next_text payload); ``expected`` = the test's expected
    # value (``None`` on the consistency / no_doctest path). Both default to
    # ``None`` so every pre-R8 trajectory (MVP / v3 / v4) parses unchanged;
    # the discriminability probe and per-source ablation read them off-line.
    out_repr: str | None = None
    expected: str | None = None


class Trajectory(BaseModel):
    instance_id: str
    source: Literal["swegym", "self_replay", "teacher_distill"]
    steps: list[Step]
