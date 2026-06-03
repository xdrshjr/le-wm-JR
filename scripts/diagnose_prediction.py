"""diagnose_prediction — lever C pre-retrain error attribution (spec §2.5).

Quantifies WHERE the per-test AUROC gap comes from before the expensive
encoder retrain: candidate-side vs test-side, plus the false-pos/false-neg
split. Writes ``attribution.json`` with a ``verdict`` that *guides* the
retrain config — NOT a hard gate (spec §2.5 P1-3): consensus_oracle already
clears SC on the same visible tests, so retrain runs. Data source (spec D2):
like ``calibrate_consensus``, reads the per-test trajectory ``test`` split
(``--traj``) + MBPP, using the trajectory's stored real-exec labels as ``M``
(no 4th exec impl — P2-5; HumanEval never read). ``P̂`` is one
``WMReranker.score_matrix`` forward; heavy deps import lazily so the shared
``error_rates`` (imported by eval_wm) stays cheap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# le-wm-JR root + repo-root scripts on sys.path (for pca.* and calibrate).
_LEWM_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LEWM_ROOT.parent
for _p in (str(_LEWM_ROOT), str(_LEWM_ROOT / "scripts"),
           str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def error_rates(probs: list, labels: list) -> dict:
    """Flat (prob, label) → Brier + false-pos / false-neg rates.

    Shared low-level aggregation (spec D3): ``eval_wm`` imports this to add the
    same fields to its per-test report. ``false_pos_rate`` is over truly-failing
    cells (predicted-pass — the consensus-breaking error); ``false_neg_rate`` is
    over truly-passing cells (predicted-fail).
    """
    n = len(probs)
    if n == 0:
        return {"brier": None, "false_pos_rate": None, "false_neg_rate": None}
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    neg = [p for p, y in zip(probs, labels) if y == 0]
    pos = [p for p, y in zip(probs, labels) if y == 1]
    return {
        "brier": round(brier, 4),
        "false_pos_rate": (round(sum(p >= 0.5 for p in neg) / len(neg), 4)
                           if neg else None),
        "false_neg_rate": (round(sum(p < 0.5 for p in pos) / len(pos), 4)
                           if pos else None),
    }


def _auroc(scores: list, labels: list):
    """Mann-Whitney rank AUROC with tie handling; ``None`` if one-class."""
    import numpy as np

    if not scores:
        return None
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    sum_pos = float(ranks[y == 1].sum())
    return round((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def _candidate_side(pred_mats: list, true_mats: list) -> dict:
    """FN within truly-correct candidates / FP within truly-wrong candidates."""
    cu_num = cu_den = wo_num = wo_den = 0
    for pm, tm in zip(pred_mats, true_mats):
        k = len(pm)
        for c in range(k):
            correct = all(v == 1 for v in tm[c])
            for p, y in zip(pm[c], tm[c]):
                if correct and y == 1:
                    cu_den += 1
                    cu_num += int(p < 0.5)
                elif (not correct) and y == 0:
                    wo_den += 1
                    wo_num += int(p >= 0.5)
    return {
        "correct_underrated": round(cu_num / cu_den, 4) if cu_den else None,
        "wrong_overrated": round(wo_num / wo_den, 4) if wo_den else None,
    }


def _test_side(pred_mats: list, true_mats: list) -> dict:
    """Per-column (assert) error rate; fraction of worst columns."""
    col_errs = []
    for pm, tm in zip(pred_mats, true_mats):
        k = len(pm)
        t = len(pm[0]) if pm else 0
        for j in range(t):
            err = sum(int((pm[c][j] >= 0.5) != (tm[c][j] == 1))
                      for c in range(k)) / max(k, 1)
            col_errs.append(err)
    mean_col = sum(col_errs) / max(len(col_errs), 1)
    worst = (sum(1 for e in col_errs if e > 1.5 * mean_col)
             / max(len(col_errs), 1))
    # MBPP-only diagnose has no doctest/no_doctest split (that distinction is
    # a HumanEval property); reported as null by construction (zero leak).
    return {"worst_cols_frac": round(worst, 4),
            "mean_col_err": round(mean_col, 4),
            "no_doctest_proposer_err": None}


def attribute_error(pred_mats: list, true_mats: list) -> dict:
    """(K,T) predicted prob mats vs (K,T) true 0/1 mats → attribution dict.

    Aggregates over the whole dev set: per-test AUROC/Brier, candidate-side
    (correct-underrated / wrong-overrated), test-side (worst columns), and the
    false-pos/neg split, then a heuristic ``verdict`` (candidate-side ⇒ encoder
    is the lever; test-side ⇒ test source has room) that guides — never blocks —
    the retrain (spec §2.5 P1-3).
    """
    flat_p, flat_y = [], []
    for pm, tm in zip(pred_mats, true_mats):
        for prow, trow in zip(pm, tm):
            flat_p.extend(prow)
            flat_y.extend(int(v) for v in trow)
    rates = error_rates(flat_p, flat_y)
    cand = _candidate_side(pred_mats, true_mats)
    test = _test_side(pred_mats, true_mats)
    cand_err = max(cand["correct_underrated"] or 0.0,
                   cand["wrong_overrated"] or 0.0)
    verdict = ("candidate-side dominates → encoder is the lever"
               if cand_err >= test["worst_cols_frac"]
               else "test-side dominates → test source has room")
    return {
        "per_test_auroc": _auroc(flat_p, flat_y),
        "brier": rates["brier"],
        "candidate_side": cand,
        "test_side": test,
        "false_pos_rate": rates["false_pos_rate"],
        "false_neg_rate": rates["false_neg_rate"],
        "verdict": verdict,
        "leak_check": "dev/MBPP only; HumanEval never read",
    }


def _pred_mats(reranker, tasks: list) -> list:
    """Predicted (K,T) prob matrix per dev task (one WM forward each)."""
    mats = []
    for task in tasks:
        p = reranker.score_matrix(task["stub"], task["programs"], task["tests"])
        mats.append(p.tolist())
    return mats


def run(args) -> int:
    from calibrate_consensus import (
        _build_reranker, _dev_tasks, _group_traj, _load_mbpp, _read_jsonl,
    )

    mbpp = _load_mbpp(args.mbpp)
    rows = _read_jsonl(Path(args.traj) / "test.jsonl")
    tasks = _dev_tasks(_group_traj(rows), mbpp)
    if not tasks:
        raise SystemExit(
            f"[diagnose] no dev tasks from {args.traj}/test.jsonl"
        )
    reranker = _build_reranker(args)
    attribution = attribute_error(
        _pred_mats(reranker, tasks), [t["label_matrix"] for t in tasks]
    )
    attribution["n_dev_tasks"] = len(tasks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(attribution, indent=2), encoding="utf-8")
    print(f"[diagnose] auroc={attribution['per_test_auroc']} "
          f"fp={attribution['false_pos_rate']} "
          f"fn={attribution['false_neg_rate']} "
          f"verdict={attribution['verdict']!r} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", required=True, help="Stage-0 ckpt to diagnose")
    ap.add_argument("--wm-config", default="wm_humaneval")
    ap.add_argument("--traj", required=True,
                    help="per-test trajectory dir (reads its test.jsonl)")
    ap.add_argument("--mbpp", default="data/benchmarks/mbpp/mbpp.jsonl")
    ap.add_argument("--out", default="runs/diag/r7_attribution.json")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
