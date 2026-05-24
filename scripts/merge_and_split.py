"""T01.8 — Merge + dedup + 80/10/10 split + SWE-bench Verified leak gate.

Inputs:  one or more JSONL files of ``Trajectory`` objects.
Outputs: ``<out>/{train,val,test}.jsonl`` and ``<out>/leak_report.json``.

Hardcoded credential policy: NONE — pure file I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from pca.data.schema import Trajectory

SWEBENCH_VERIFIED_HF = "princeton-nlp/SWE-bench_Verified"


def _load_verified_ids() -> set[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(SWEBENCH_VERIFIED_HF, split="test")
        return {str(row["instance_id"]) for row in ds}
    except Exception as exc:
        print(
            f"[merge_and_split] WARN: could not load Verified ids ({exc}); "
            "leak gate degraded to empty set"
        )
        return set()


def _iter_trajectories(paths: list[Path]):
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                traj = Trajectory.model_validate_json(line)
                key = hashlib.sha256(
                    f"{traj.instance_id}|{len(traj.steps)}".encode()
                ).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                yield traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="raw JSONL files")
    parser.add_argument(
        "--out", default="data/trajectories/v1", help="output directory"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split", nargs=3, type=float, default=(0.8, 0.1, 0.1)
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verified_ids = _load_verified_ids()
    leaks: list[str] = []

    all_trajs: list[Trajectory] = []
    for traj in _iter_trajectories([Path(p) for p in args.inputs]):
        if traj.instance_id in verified_ids:
            leaks.append(traj.instance_id)
            continue
        all_trajs.append(traj)

    rng = random.Random(args.seed)
    rng.shuffle(all_trajs)

    n = len(all_trajs)
    n_train = int(n * args.split[0])
    n_val = int(n * args.split[1])
    splits = {
        "train": all_trajs[:n_train],
        "val": all_trajs[n_train : n_train + n_val],
        "test": all_trajs[n_train + n_val :],
    }

    for name, items in splits.items():
        out_path = out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for traj in items:
                fh.write(traj.model_dump_json() + "\n")
        print(f"[merge_and_split] {name}: {len(items)} → {out_path}")

    leak_report = {
        "verified_ids_loaded": len(verified_ids),
        "leak_count": len(leaks),
        "leak_instance_ids": leaks[:50],
        "method": "instance_id exact match",
    }
    with (out_dir / "leak_report.json").open("w", encoding="utf-8") as fh:
        json.dump(leak_report, fh, indent=2)
    if leaks:
        raise SystemExit(
            f"[merge_and_split] HARD GATE: {len(leaks)} verified-set leaks"
        )


if __name__ == "__main__":
    main()
