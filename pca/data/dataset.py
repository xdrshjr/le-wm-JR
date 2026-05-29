"""TrajectoryDataset — JSONL backend (HDF5 deferred to v2, R08)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pydantic import TypeAdapter
from torch.utils.data import Dataset

from pca.action.schema import ExecutableOp
from pca.data.schema import Trajectory

_TRAJ_ADAPTER: TypeAdapter[Trajectory] = TypeAdapter(Trajectory)


@dataclass
class TrajectoryDatasetConfig:
    path: str
    split: str = "train"
    history_size: int = 3
    num_preds: int = 1
    seed: int = 0
    max_obs_chars: int = 16000


class TrajectoryDataset(Dataset):
    """Yields fixed-length windows (history_size + num_preds) of
    ``(obs_text, op)`` pairs from a JSONL trajectory file.

    The on-disk format is one ``Trajectory`` JSON per line under
    ``<path>/<split>.jsonl``. Windows are flattened lazily so that
    every trajectory contributes ``max(0, len(steps) - T + 1)`` samples.
    """

    def __init__(self, cfg: TrajectoryDatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.window = cfg.history_size + cfg.num_preds
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._trajectories: list[Trajectory] = []
        self._index: list[tuple[int, int]] = []

        path = Path(cfg.path) / f"{cfg.split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"trajectory jsonl not found: {path} "
                "(run scripts/collect_swegym.py first)"
            )
        self._load(path)

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                traj = _TRAJ_ADAPTER.validate_python(json.loads(line))
                if len(traj.steps) < self.window:
                    continue
                tid = len(self._trajectories)
                self._trajectories.append(traj)
                num_windows = len(traj.steps) - self.window + 1
                for offset in range(num_windows):
                    self._index.append((tid, offset))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        tid, offset = self._index[idx]
        traj = self._trajectories[tid]
        steps = traj.steps[offset : offset + self.window]
        cap = self.cfg.max_obs_chars
        obs_text: list[str] = [s.obs_text[:cap] for s in steps]
        op_list: list[ExecutableOp] = [s.op for s in steps]
        # Window label = step0's label (the predicted-from step). This is
        # the only entry point for ``label`` into the batch (spec §F1); the
        # OutcomeHead is supervised on the post-``op`` outcome of step0.
        return {
            "obs_text": obs_text,
            "op": op_list,
            "instance_id": traj.instance_id,
            "label": steps[0].label,
        }


@dataclass
class HDF5BackendConfig:
    path: str
    split: str = "train"
    extra: dict = field(default_factory=dict)


class HDF5Backend:
    """Deferred to adaptation-spec v2 (R08).

    Schema reconciliation against ``eval.HDF5Dataset`` is tracked there.
    """

    def __init__(self, cfg: HDF5BackendConfig) -> None:
        raise NotImplementedError(
            "HDF5 backend deferred to adaptation-spec v2; use JSONL"
        )
