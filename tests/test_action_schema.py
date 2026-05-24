"""Pydantic round-trip tests for the 8 op types.

Runnable as ``python tests/test_action_schema.py`` (CLAUDE.md — no
formal pytest suite required for research code).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import TypeAdapter, ValidationError  # noqa: E402

from pca.action.schema import OP_TYPES, ExecutableOp  # noqa: E402


_ADAPTER: TypeAdapter[ExecutableOp] = TypeAdapter(ExecutableOp)


GOLDEN = [
    {"op_type": "apply_patch", "diff": "---\n+++\n"},
    {"op_type": "run_test", "selector": "tests/", "timeout_sec": 60},
    {"op_type": "search_code", "query": "TODO", "path": "src/"},
    {"op_type": "edit_file", "path": "a.py", "start_line": 1, "new_text": "x"},
    {"op_type": "read_file", "path": "a.py"},
    {"op_type": "ls", "path": "."},
    {"op_type": "cd", "path": ".."},
    {"op_type": "git_diff", "ref": "HEAD"},
]


def _round_trip(op_dict: dict) -> None:
    op = _ADAPTER.validate_python(op_dict)
    dumped = json.loads(op.model_dump_json())
    again = _ADAPTER.validate_python(dumped)
    assert op == again, f"round-trip mismatch: {op_dict}"


def main() -> int:
    seen = set()
    for op_dict in GOLDEN:
        _round_trip(op_dict)
        seen.add(op_dict["op_type"])

    missing = set(OP_TYPES) - seen
    if missing:
        print(f"FAIL: missing op_type coverage: {missing}")
        return 1

    try:
        _ADAPTER.validate_python({"op_type": "not_a_real_op"})
    except ValidationError:
        pass
    else:
        print("FAIL: invalid op_type was accepted")
        return 1

    print(f"OK: {len(GOLDEN)} ops round-tripped, 8/8 op_types covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
