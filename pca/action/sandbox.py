"""Sandbox interface — Docker/nsjail implementation deferred to v2.

Phase B'.3 (T03.8) from
``docs/plans/world-model-llm-coding-fusion/todos/03-action-space.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pca.action.schema import ExecutableOp


@dataclass
class Observation:
    text: str
    exit_code: int = 0
    cwd: str = "."


@runtime_checkable
class Sandbox(Protocol):
    """Minimal sandbox protocol used by PCA inference loop.

    Concrete Docker / nsjail backends land in adaptation-spec v2.
    """

    def reset(self) -> Observation: ...

    def step(self, op: ExecutableOp) -> Observation: ...

    def snapshot(self) -> str: ...

    def restore(self, snapshot_id: str) -> Observation: ...
