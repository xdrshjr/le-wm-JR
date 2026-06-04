"""PEC consensus — zero-execution CodeT over a predicted pass matrix.

The world model predicts, for every ``(candidate, visible test)`` pair, the
probability that the candidate passes the test (``WMReranker.score_matrix``,
a ``(K, T)`` matrix produced by a forward pass — **no candidate is ever
executed**). ``consensus_rank`` then reranks candidates with CodeT-style
dual consistency (Chen et al., 2022) on that matrix, so a single mistaken
prediction is averaged out by agreement across candidates and tests.

Three modes (spec §2.2(c) / §2.4):
  soft (default, R2-fixed): ``score(c) = (1/T)·Σ_t P[c,t]·Σ_c' P[c',t]`` —
      PASS-mass agreement only. A candidate scores high iff the tests it is
      predicted to pass are also predicted-passed by many other candidates.
      The ``(1−P)(1−P')`` "shared failure" reward of the v1 draft is
      **deliberately removed**: rewarding mutually-all-failing candidates is
      anti-CodeT and reproduces the round-5 "ranking anti-correlated with
      correctness" failure (spec R2). A candidate predicted to fail every
      test (P≈0) scores ≈0 and stays last.
  soft_conf (round-7, spec §2.4 lever B): confidence-weighted soft. Each
      prediction is weighted by ``w[c,t] = (2·|P[c,t]−0.5|)^γ`` so a
      P≈0.5 (uncertain) prediction barely moves the consensus while a
      P≈0.95 (confident) one drives it — predicted-pass noise stops
      shattering the consensus signal. Same R2 invariant: only shared
      *pass* mass is rewarded (no shared-failure term). An all-uncertain
      matrix (total weight ≈ 0) falls back to log-probs.
  hard (ablation): CodeT original ``|S| × |y|`` on ``B = ⟦P ≥ theta⟧``,
      where ``|S|`` is the size of the consensus set sharing a candidate's
      pass signature and ``|y|`` its predicted-pass count.

Honesty invariant (spec §6): this module only consumes a prediction matrix
and reorders candidates — it never executes anything.
"""
from __future__ import annotations

import ast
import math
from collections import Counter


def _to_rows(matrix) -> list[list[float]]:
    """Coerce a ``(K, T)`` tensor / nested sequence to ``list[list[float]]``."""
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [list(row) for row in matrix]


def _soft_scores(rows: list[list[float]]) -> list[float]:
    """PASS-mass agreement (R2-fixed): score[c] = mean_t P[c,t]·Σ_c' P[c',t]."""
    k = len(rows)
    t = len(rows[0])
    colsum = [sum(rows[c][j] for c in range(k)) for j in range(t)]
    return [
        sum(rows[c][j] * colsum[j] for j in range(t)) / t
        for c in range(k)
    ]


def _soft_conf_scores(
    rows: list[list[float]], gamma: float, logprobs,
) -> list[float]:
    """Confidence-weighted PASS-mass agreement (R2-fixed; spec §2.4).

    ``w[c,t] = (2|P[c,t]-0.5|)^gamma`` ∈ [0,1] is each prediction's
    confidence; the score is the soft consensus reweighted by it::

        score(c) = (1/(Σ_t w[c,t]+ε))·Σ_t w[c,t]·P[c,t]·(Σ_c' w[c',t]·P[c',t])

    Only shared *pass* mass is rewarded (守 R2 — no shared-failure term). A
    matrix of all-uncertain predictions (total weight ≈ 0) degenerates to a
    log-prob fallback, matching the ``T == 0`` contract.
    """
    k = len(rows)
    t = len(rows[0])
    eps = 1e-9
    w = [
        [(2.0 * abs(rows[c][j] - 0.5)) ** gamma for j in range(t)]
        for c in range(k)
    ]
    if sum(w[c][j] for c in range(k) for j in range(t)) < eps:
        return list(logprobs) if logprobs is not None else [0.0] * k
    wcol = [sum(w[c][j] * rows[c][j] for c in range(k)) for j in range(t)]
    scores = []
    for c in range(k):
        denom = sum(w[c][j] for j in range(t)) + eps
        num = sum(w[c][j] * rows[c][j] * wcol[j] for j in range(t))
        scores.append(num / denom)
    return scores


def _hard_scores(rows: list[list[float]], theta: float) -> list[float]:
    """CodeT original ``|S| × |y|`` on the thresholded pass matrix."""
    t = len(rows[0])
    sigs = [
        tuple(1 if rows[c][j] >= theta else 0 for j in range(t))
        for c in range(len(rows))
    ]
    counts = Counter(sigs)
    return [float(counts[sig] * sum(sig)) for sig in sigs]


def consensus_rank(
    matrix, logprobs, *, theta: float = 0.5, mode: str = "soft",
    gamma: float = 1.0,
) -> list[float]:
    """CodeT consensus score per candidate from a ``(K, T)`` prob matrix.

    Returns a per-candidate score list (higher = better); the caller takes
    the argmax and breaks ties by ``logprobs`` (spec §4.2). When ``T == 0``
    the caller is expected to skip this call, but as a safety net the raw
    ``logprobs`` are returned so a degenerate matrix never crashes ranking.

    ``mode="soft_conf"`` (round-7, spec §2.4) confidence-weights the soft
    consensus by ``gamma``; ``soft``/``hard`` ignore ``gamma`` and stay
    byte-identical to the round-6 behaviour (default ``gamma=1.0`` → zero
    regression for existing callers).
    """
    rows = _to_rows(matrix)
    if not rows:
        return []
    t = len(rows[0])
    if t == 0:
        return list(logprobs) if logprobs is not None else [0.0] * len(rows)
    if mode == "hard":
        return _hard_scores(rows, theta)
    if mode == "soft_conf":
        return _soft_conf_scores(rows, gamma, logprobs)
    return _soft_scores(rows)


# ----- round-8 execution-derived pass matrix (spec §2.4.1) -------------


def parse_assert_io(assert_text: str) -> tuple[str | None, str | None]:
    """Split an assert/doctest into ``(call_form, expected)`` (spec §2.4.1 C-3).

    ``assert f(2, 3) == 5`` → ``("f(2, 3)", "5")``. When the assert is not an
    ``==`` comparison (``assert is_prime(7)``) the call form is the whole test
    expression and ``expected`` is ``None`` → the consistency path. Parse
    failures return ``(None, None)`` so the caller can drop the test.
    """
    try:
        tree = ast.parse(assert_text.strip())
    except SyntaxError:
        return None, None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None, None
    test = tree.body[0].test
    if (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return ast.unparse(test.left), ast.unparse(test.comparators[0])
    return ast.unparse(test), None


def _to_list3(x) -> list[list[list[float]]]:
    """Coerce a (K, T, P) tensor / nested sequence to nested float lists."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [[[float(v) for v in cell] for cell in row] for row in x]


def _to_list2(x) -> list[list[float]]:
    """Coerce a (T, P) tensor / nested sequence to nested float lists."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [[float(v) for v in row] for row in x]


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _expected_matrix(out3, exp2, tau: float) -> list[list[float]]:
    """has_doctest path: P[c,t] = σ(cos(ô[c,t], z(e_t)) / τ) (spec §2.4.1)."""
    k = len(out3)
    t = len(out3[0]) if k else 0
    return [
        [_sigmoid(_cos(out3[c][j], exp2[j]) / tau) for j in range(t)]
        for c in range(k)
    ]


def _consistency_matrix(
    out3, tau: float, cluster_thr: float
) -> list[list[float]]:
    """no_doctest path: soft majority-cluster membership of ô[·,t] (MBR-EXEC).

    For each test column, ``P[c,t]`` is the mean soft agreement of candidate
    ``c``'s predicted output with every candidate's (incl. itself), so a
    candidate whose output is shared by many others scores high — the
    execution analogue of self-consistency, fed to the same ``consensus_rank``
    (spec §2.4.1, no expected value needed).
    """
    k = len(out3)
    t = len(out3[0]) if k else 0
    matrix = [[0.0] * t for _ in range(k)]
    for j in range(t):
        outs = [out3[c][j] for c in range(k)]
        for c in range(k):
            agree = sum(
                _sigmoid((_cos(outs[c], outs[d]) - cluster_thr) / tau)
                for d in range(k)
            )
            matrix[c][j] = agree / max(k, 1)
    return matrix


def exec_pass_from_outputs(
    out_embeds, expected_embeds=None, *,
    tau: float = 1.0, cluster_thr: float = 0.7,
) -> list[list[float]]:
    """(K, T, P) predicted output embeddings → (K, T) pass/consistency matrix.

    The single source of the round-8 PEC matrix (``score_matrix_exec`` /
    ``AlignedWMLLM.predict_pass_matrix`` are thin wrappers; spec §2.4.1 C-8).
    ``expected_embeds`` given (has_doctest, (T, P)) → compare predicted output
    to the expected output; ``None`` (no_doctest) → candidate-vs-candidate
    output consistency. The result feeds the *unchanged* ``consensus_rank``;
    nothing is executed.
    """
    out3 = _to_list3(out_embeds)
    if not out3 or not out3[0]:
        return [[] for _ in out3]
    tau = max(float(tau), 1e-3)
    if expected_embeds is not None:
        return _expected_matrix(out3, _to_list2(expected_embeds), tau)
    return _consistency_matrix(out3, tau, cluster_thr)


def _extract_doctests(prompt: str, entry_point: str) -> list[str]:
    """Lazily reuse ``bench_humaneval.extract_doctest_examples`` (spec §2.2a).

    Imported on demand (and the repo-root ``scripts/`` added to ``sys.path``)
    to avoid a library→script import cycle at module load. Callers that
    already extracted the doctest asserts should pass them via
    ``doctest_asserts`` and skip this path entirely.
    """
    try:
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[3] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from bench_humaneval import extract_doctest_examples
    except Exception:
        return []
    return extract_doctest_examples(prompt, entry_point)


def gather_visible_tests(
    prompt: str, entry_point: str, proposer=None,
    *, doctest_asserts: list[str] | None = None, max_tests: int = 3,
) -> list[str]:
    """Collect the visible test set for one problem (spec §2.2(a)).

    ``has_doctest`` problems use the docstring ``>>>`` examples (passed in
    via ``doctest_asserts`` by the bench, else lazily extracted);
    ``no_doctest`` problems fall back to ≤``max_tests`` LLM self-proposed
    asserts (cached in ``proposer``). The only difference between the two
    subsets is the test *source* — both then run the same PEC algorithm.
    Returns ``[]`` if neither source yields a test (caller → logprob argmax).
    """
    if doctest_asserts is None:
        doctest_asserts = _extract_doctests(prompt, entry_point)
    if doctest_asserts:
        return list(doctest_asserts)
    if proposer is None:
        return []
    return list(proposer.propose(prompt))[:max_tests]
