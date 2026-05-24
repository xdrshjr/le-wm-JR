"""Variable-length collation for trajectory batches.

The dataset yields python lists (not tensors) — actual tensorization
happens inside ``TextJEPA.encode`` so that the encoder owns the
tokenizer. ``collate_trajectories`` keeps batch dimension first
and preserves per-step structure ``(B, T)``.
"""
from __future__ import annotations

from typing import Iterable

from pca.action.schema import ExecutableOp


def collate_trajectories(batch: Iterable[dict]) -> dict:
    items = list(batch)
    if not items:
        return {"obs_text": [], "op": [], "instance_id": []}

    obs_text: list[list[str]] = [item["obs_text"] for item in items]
    ops: list[list[ExecutableOp]] = [item["op"] for item in items]
    instance_ids: list[str] = [item["instance_id"] for item in items]

    return {
        "obs_text": obs_text,
        "op": ops,
        "instance_id": instance_ids,
    }
