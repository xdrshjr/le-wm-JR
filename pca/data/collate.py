"""Variable-length collation for trajectory batches.

The dataset yields python lists (not tensors) — actual tensorization
happens inside ``TextJEPA.encode`` so that the encoder owns the
tokenizer. ``collate_trajectories`` keeps batch dimension first
and preserves per-step structure ``(B, T)``.
"""
from __future__ import annotations

from typing import Iterable

import torch

from pca.action.schema import ExecutableOp


def _collate_labels(items: list[dict]) -> "torch.Tensor | None":
    """Stack per-window labels into ``(B,)``; ``None`` if any are missing
    so unlabeled (MVP / synthetic) batches short-circuit the BCE branch
    in ``pca_forward`` (spec §F1, §6.4)."""
    labels = [item.get("label") for item in items]
    if any(v is None for v in labels):
        return None
    return torch.tensor(labels, dtype=torch.float32)


def collate_trajectories(batch: Iterable[dict]) -> dict:
    items = list(batch)
    if not items:
        return {"obs_text": [], "op": [], "instance_id": [], "label": None}

    obs_text: list[list[str]] = [item["obs_text"] for item in items]
    ops: list[list[ExecutableOp]] = [item["op"] for item in items]
    instance_ids: list[str] = [item["instance_id"] for item in items]

    return {
        "obs_text": obs_text,
        "op": ops,
        "instance_id": instance_ids,
        "label": _collate_labels(items),
    }
