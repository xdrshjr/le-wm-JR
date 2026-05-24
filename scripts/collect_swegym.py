"""T01.5 — Pull SWE-Gym → unified trajectory JSONL (Spec 01).

Reads HuggingFace ``datasets`` (use ``HF_ENDPOINT=https://hf-mirror.com``
when blocked — CLAUDE.md). Writes one ``Trajectory`` JSON per line to
``<out>/raw_swegym.jsonl``. Splits + manifest happen in
``merge_and_split.py`` / ``make_manifest.py``.

Credentials policy: this script reads ALL SSH creds from environment
variables (``LEWM_SSH_*``); no hardcoded host or password — see
``CLAUDE.md`` §Security and ``tests/test_no_credential_leak.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pca.action.compiler import NLIntentCompiler, NeedsClarification
from pca.action.schema import ApplyPatchArgs
from pca.data.schema import Step, Trajectory


def _row_to_trajectory(row: dict, compiler: NLIntentCompiler) -> Trajectory | None:
    instance_id = row.get("instance_id") or row.get("id") or ""
    if not instance_id:
        return None

    steps: list[Step] = []
    raw_steps = row.get("trajectory") or row.get("steps") or []
    for raw in raw_steps:
        obs_text = raw.get("observation") or raw.get("obs", "")
        intent = raw.get("action") or raw.get("intent", "")
        if not obs_text or not intent:
            continue
        op = compiler.compile(intent)
        if isinstance(op, NeedsClarification):
            patch = raw.get("patch") or raw.get("model_patch")
            if patch:
                op = ApplyPatchArgs(diff=patch)
            else:
                continue
        steps.append(
            Step(
                obs_text=obs_text,
                op=op,
                obs_next_text=raw.get("next_observation"),
                expert_intent=intent,
            )
        )

    if not steps:
        return None
    return Trajectory(
        instance_id=str(instance_id), source="swegym", steps=steps
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="SWE-Gym/SWE-Gym",
        help="HuggingFace dataset id",
    )
    parser.add_argument(
        "--split", default="train", help="dataset split to pull"
    )
    parser.add_argument(
        "--out",
        default="data/trajectories/raw",
        help="output directory",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="0 = all rows"
    )
    args = parser.parse_args()

    if os.environ.get("HF_ENDPOINT") is None:
        print(
            "[collect_swegym] HF_ENDPOINT unset; if you are on the remote box, "
            "export HF_ENDPOINT=https://hf-mirror.com first (CLAUDE.md)."
        )

    from datasets import load_dataset

    ds = load_dataset(args.hf_repo, split=args.split, streaming=True)
    compiler = NLIntentCompiler()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_swegym.jsonl"

    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            traj = _row_to_trajectory(row, compiler)
            if traj is None:
                continue
            fh.write(traj.model_dump_json() + "\n")
            n_written += 1
            if n_written % 500 == 0:
                print(f"[collect_swegym] wrote {n_written} trajectories")

    print(f"[collect_swegym] done — {n_written} → {out_path}")


if __name__ == "__main__":
    main()
