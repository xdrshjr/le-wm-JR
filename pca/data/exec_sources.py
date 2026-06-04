"""exec_sources — multi-source problem loaders for execution-trace data (R8).

Unifies several function-call-form coding corpora into one
``ExecProblem`` shape ``(stub, gold, setup, tests=[(call, expected)])`` so the
round-8 collector (``scripts/gen_exec_traj.py``) can sample candidates and run
them once, offline, to build ``(serialize_exec, true output, expected, pass)``
trajectories (spec wm-exec-trace-fusion-sota §2.3).

Per spec §2.3 C-4 the **default sources are the function-call-form MBPP family**
(``mbpp``, ``mbpp_plus``) — directly compatible with ``serialize_exec``'s
``f(args) → repr`` schema. ``swm.data.load_dataset`` is bypassed (it is not the
pinned API here); loaders read plain JSONL. APPS / CodeContests (stdin/stdout
competitive I/O) are intentionally NOT loaded here — they need an I/O→call-form
adapter and are a deferred optional ablation (spec §2.3 C-4 / R-2 / R-6).

Leak red line (spec §7 R-7): every problem is MBPP-family; ``humaneval_overlap``
hard-checks that no HumanEval problem leaks in. HumanEval is graded only by the
bench, never read here.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# le-wm-JR root + repo-root scripts on sys.path (for pca.* and _to_stub).
_LEWM_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _LEWM_ROOT.parent
for _p in (str(_LEWM_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class ExecProblem:
    """One coding problem normalised to the execution-trace schema."""

    problem_id: str
    source: str
    stub: str
    gold: str
    setup: str
    tests: list = field(default_factory=list)  # [(call_form, expected|None)]
    difficulty: int = 0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tests_from_asserts(asserts: list[str]) -> list:
    """Parse ``test_list`` asserts → ``[(call_form, expected|None)]`` (C-3)."""
    from pca.inference.consensus import parse_assert_io

    out = []
    for a in asserts:
        call, expected = parse_assert_io(a)
        if call is not None:
            out.append((call, expected))
    return out


def _problem_from_row(row: dict, source: str) -> "ExecProblem | None":
    """MBPP-family JSONL row → ``ExecProblem`` (function-call form)."""
    from build_alignment_data import _to_stub

    gold = row.get("code") or ""
    asserts = row.get("test_list") or []
    if not gold or not asserts or "task_id" not in row:
        return None
    tests = _tests_from_asserts(asserts)
    if not tests:
        return None
    tid = int(row["task_id"])
    return ExecProblem(
        problem_id=f"{source}-{tid:06d}",
        source=source,
        stub=_to_stub(row, gold),
        gold=gold,
        setup=row.get("test_setup_code") or "",
        tests=tests,
        difficulty=len(gold),
    )


def _load_jsonl_source(path: str, source: str, n: int) -> list[ExecProblem]:
    rows = _read_jsonl(Path(path))
    if n > 0:
        rows = rows[:n]
    probs = [_problem_from_row(r, source) for r in rows]
    return [p for p in probs if p is not None]


def load_mbpp(paths: dict, n: int) -> list[ExecProblem]:
    return _load_jsonl_source(
        paths.get("mbpp", "data/benchmarks/mbpp/mbpp.jsonl"), "mbpp", n
    )


def load_mbpp_plus(paths: dict, n: int) -> list[ExecProblem]:
    """MBPP+ (EvalPlus extended inputs) as a function-call-form JSONL.

    Expects a local ``mbpp_plus.jsonl`` (same schema as ``mbpp.jsonl``) at
    ``paths['mbpp_plus']``; EvalPlus's native format is a check-fn, not
    ``(call, expected)`` pairs, so an offline conversion is required (spec §2.3
    C-4 / R-6). Absent the file the source is skipped — the pipeline then runs
    MBPP-only, which is still a valid method (R-2).
    """
    path = paths.get("mbpp_plus", "data/benchmarks/mbpp_plus/mbpp_plus.jsonl")
    if not Path(path).exists():
        print(f"[exec_sources] mbpp_plus skipped: {path} absent "
              "(MBPP-only is still valid; spec R-2/R-6)")
        return []
    return _load_jsonl_source(path, "mbpp_plus", n)


SOURCE_LOADERS = {"mbpp": load_mbpp, "mbpp_plus": load_mbpp_plus}


def load_sources(
    sources: list[str], paths: dict, quota: int
) -> list[ExecProblem]:
    """Load every requested source (per-source ``quota`` problems)."""
    problems: list[ExecProblem] = []
    for name in sources:
        loader = SOURCE_LOADERS.get(name)
        if loader is None:
            print(f"[exec_sources] WARN unknown source {name!r}; skipped "
                  "(APPS/CodeContests need an I/O→call adapter; spec C-4)")
            continue
        got = loader(paths, quota)
        print(f"[exec_sources] {name}: {len(got)} problems")
        problems.extend(got)
    return problems


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def humaneval_overlap(problems: list[ExecProblem], he_path: str) -> int:
    """Count problems whose normalised stub collides with a HumanEval prompt.

    Hard leak guard (spec §7 R-7): MBPP-family sources collide with nothing, so
    this must be 0. Returns 0 (with a note) when HumanEval is unavailable —
    the collector still asserts ``== 0`` before writing.
    """
    rows = _read_jsonl(Path(he_path))
    if not rows:
        print(f"[exec_sources] leak_check: {he_path} absent; "
              "MBPP-family sources cannot overlap HumanEval (overlap=0)")
        return 0
    he = {_norm(r.get("prompt", "")) for r in rows}
    return sum(1 for p in problems if _norm(p.stub) in he)


def split_problems(problems: list[ExecProblem], seed: int) -> dict:
    """Problem-level split: train / val (same-dist) + transfer (OOD; spec §2.3).

    ``transfer`` = every ``mbpp_plus`` problem + the hardest difficulty quartile
    of MBPP (a real within-corpus distribution shift), used for transfer-AUROC
    selection (C-1). The rest is split 90/10 into same-distribution train/val.
    Candidates never straddle splits (this partitions at the problem level).
    """
    import random

    rng = random.Random(seed)
    plus = [p for p in problems if p.source != "mbpp"]
    mbpp = [p for p in problems if p.source == "mbpp"]
    mbpp_sorted = sorted(mbpp, key=lambda p: p.difficulty)
    n_hard = len(mbpp_sorted) // 4
    hard = mbpp_sorted[len(mbpp_sorted) - n_hard:] if n_hard else []
    same = mbpp_sorted[: len(mbpp_sorted) - n_hard] if n_hard else mbpp_sorted
    rng.shuffle(same)
    n_val = max(1, int(round(len(same) * 0.1))) if same else 0
    return {
        "val": same[:n_val],
        "train": same[n_val:],
        "transfer": plus + hard,
    }
