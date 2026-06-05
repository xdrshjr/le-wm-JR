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
    seed_id: str = ""  # heval_style: source MBPP task (zero-padded tail)


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


def load_heval_style(paths: dict, n: int) -> list[ExecProblem]:
    """HumanEval-style synthetic problems (R9 spec §2.3 B-2).

    Reads ``gen_heval_style.py``'s output (``problem_id/stub/gold/setup/
    tests/style`` rows — gold-verified, 8-gram-deduped against HumanEval).
    Absent file → skipped with a note (the mbpp[,mbpp_plus] pipeline is
    still valid; the missing target-style source is a G1-amber signal).
    """
    path = Path(paths.get(
        "heval_style", "data/benchmarks/heval_style/heval_style.jsonl"
    ))
    rows = _read_jsonl(path)
    if not rows:
        print(f"[exec_sources] heval_style skipped: {path} absent/empty "
              "(run scripts/gen_heval_style.py; spec §2.3 B-2)")
        return []
    if n > 0:
        rows = rows[:n]
    probs = []
    for row in rows:
        stub, gold = row.get("stub") or "", row.get("gold") or ""
        tests = [tuple(t) for t in (row.get("tests") or []) if t]
        if not stub or not tests:
            continue
        probs.append(ExecProblem(
            problem_id=row.get("problem_id") or f"heval_style-{len(probs):06d}",
            source="heval_style",
            stub=stub,
            gold=gold,
            setup=row.get("setup") or "",
            tests=tests,
            difficulty=len(gold),
            seed_id=_seed_tail(row.get("seed_task_id")),
        ))
    return probs


def _seed_tail(seed) -> str:
    """``seed_task_id`` → zero-padded tail matching ``_tid_tail`` — the
    seed-level split-leak guard key in ``split_problems`` (review fix R9)."""
    try:
        return f"{int(seed):06d}"
    except (TypeError, ValueError):
        return ""


SOURCE_LOADERS = {
    "mbpp": load_mbpp,
    "mbpp_plus": load_mbpp_plus,
    "heval_style": load_heval_style,
}


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


# ----- 8-gram similarity (R9 leak red line, spec §2.3 B-2 / §7 R-2) -----

def _ngram_set(text: str, n: int = 8) -> set:
    t = _norm(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard_8gram(a: str, b: str) -> float:
    """Char 8-gram Jaccard similarity — the heval_style dedup metric.

    Shared by ``gen_heval_style`` (collection-time hard filter) and the
    collector's double hard check (``heval_style_max_sim < sim_thr``).
    """
    sa, sb = _ngram_set(a), _ngram_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def heval_style_max_sim(problems: list[ExecProblem], he_path: str) -> float:
    """Max 8-gram Jaccard of any heval_style stub vs a HumanEval prompt.

    The second hard leak check (with ``humaneval_overlap == 0``): the
    collector asserts ``< sim_thr`` before writing (spec §2.3 B-3).
    Returns 0.0 when HumanEval or the source is absent (noted upstream;
    the COLLECTOR hard-fails on a missing HumanEval file — review cond. 6).
    UNROUNDED — rounding before the ``>= sim_thr`` compare turns a kept
    0.34996 item into a spurious 0.35 gate trip (review fix R9); callers
    round for display/manifest only.
    """
    stubs = [p.stub for p in problems if p.source == "heval_style"]
    rows = _read_jsonl(Path(he_path))
    if not stubs or not rows:
        return 0.0
    he_sets = [_ngram_set(r.get("prompt", "")) for r in rows]
    worst = 0.0
    for stub in stubs:
        s = _ngram_set(stub)
        if not s:
            continue
        for h in he_sets:
            if not h:
                continue
            sim = len(s & h) / len(s | h)
            worst = max(worst, sim)
    return worst


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


def _tid_tail(problem_id: str) -> str:
    """``mbpp_plus-000123`` → ``000123`` (numeric tail for overlap checks)."""
    return problem_id.rsplit("-", 1)[-1]


def _split_heval(heval: list, hard_tids: set, mbpp_tids: set, rng) -> tuple:
    """heval_style → (holdout→transfer, →train), SEED-task aware (R9 fix):
    a style-rewrite is a semantic near-duplicate of its seed MBPP task, so
    (a) rewrites of hard-quartile (transfer) seeds must follow the seed
    into transfer, and (b) the transfer holdout is drawn ONLY from rewrites
    whose seed task was never loaded — else transfer is contaminated by
    near-duplicates of trained problems and the gate metric inflates."""
    hv_hard, hv_fresh, hv_seen = [], [], []
    for p in heval:
        if p.seed_id and p.seed_id in hard_tids:
            hv_hard.append(p)
        elif p.seed_id and p.seed_id not in mbpp_tids:
            hv_fresh.append(p)
        else:  # seed trained, or legacy row without seed_task_id
            hv_seen.append(p)
    rng.shuffle(hv_fresh)
    want = max(0, int(round(len(heval) * 0.25)) - len(hv_hard))
    if want > len(hv_fresh):
        print(f"[exec_sources] NOTE: heval_style holdout {len(hv_fresh)}"
              f"+{len(hv_hard)} < 25% target — NOT padded with trained-seed"
              " rewrites (transfer purity > holdout size; R9 fix)")
    return hv_hard + hv_fresh[:want], hv_seen + hv_fresh[want:]


def split_problems(problems: list[ExecProblem], seed: int) -> dict:
    """Problem-level split: train / val (same-dist) + transfer (real OOD).

    R9 upgrade (spec §2.3 B-3): ``transfer`` = heval_style holdout (the
    TARGET style; seed-task aware, see ``_split_heval``) ∪ mbpp_plus-only
    problems (task_ids absent from the loaded mbpp set) ∪ the hardest MBPP
    difficulty quartile. mbpp_plus rows sharing an mbpp task_id join the
    pool of THEIR task's split — train normally, transfer when the task is
    in the hard quartile (same-task extended inputs must never straddle
    the train/transfer boundary; review fix R9). The remaining pool splits
    90/10 into same-distribution train/val. Candidates never straddle
    splits (problem-level partition).
    """
    import random

    rng = random.Random(seed)
    mbpp = [p for p in problems if p.source == "mbpp"]
    plus = [p for p in problems if p.source == "mbpp_plus"]
    heval = [p for p in problems if p.source == "heval_style"]
    other = [p for p in problems
             if p.source not in ("mbpp", "mbpp_plus", "heval_style")]
    mbpp_tids = {_tid_tail(p.problem_id) for p in mbpp}
    plus_only = [p for p in plus if _tid_tail(p.problem_id) not in mbpp_tids]
    plus_shared = [p for p in plus if _tid_tail(p.problem_id) in mbpp_tids]
    mbpp_sorted = sorted(mbpp, key=lambda p: p.difficulty)
    n_hard = len(mbpp_sorted) // 4
    hard = mbpp_sorted[len(mbpp_sorted) - n_hard:] if n_hard else []
    easy = mbpp_sorted[: len(mbpp_sorted) - n_hard] if n_hard else mbpp_sorted
    hard_tids = {_tid_tail(p.problem_id) for p in hard}
    plus_train = [p for p in plus_shared
                  if _tid_tail(p.problem_id) not in hard_tids]
    plus_hard = [p for p in plus_shared
                 if _tid_tail(p.problem_id) in hard_tids]
    heval_hold, heval_train = _split_heval(heval, hard_tids, mbpp_tids, rng)
    same = easy + plus_train + heval_train + other
    rng.shuffle(same)
    n_val = max(1, int(round(len(same) * 0.1))) if same else 0
    return {
        "val": same[:n_val],
        "train": same[n_val:],
        "transfer": heval_hold + plus_only + plus_hard + hard,
    }
