"""calibrate_consensus — independent consensus calibration (spec §2.6, R1/R6).

Headline-path calibration that is fully decoupled from the aligned model
(``calibrate_verifier`` is kept only for the ``fused``/``pca_align``
ablations). It needs ONLY a ``WMReranker`` (verifier head) + the per-test
trajectory ``test`` split + MBPP, and writes ``consensus_config.json`` next
to the Stage-0 checkpoint for the bench ``--wm-score consensus`` path to read.

What it does (all on MBPP / held-out dev — **HumanEval is never read**):
  1. group the per-test ``test`` split by ``mbpp-<tid>-cand<c>-t<t>`` into one
     program per candidate, with a per-(candidate,test) binary label matrix and
     a per-candidate correctness label = logical AND over its tests (R6);
  2. retrieve the visible assert text from MBPP ``test_list`` by ``task_id``
     (zero new execution, zero leakage; same source the trajectories used);
  3. fit the verifier temperature by minimising per-test NLL (report ECE);
  4. grid-search ``theta × mode`` by dev rerank accuracy, with the
     non-destructive guardrail: if the best consensus config does not strictly
     beat the mean-probability ("plain verifier") baseline, fall back to
     ``mode=soft, theta=0.5, w_l=0`` (spec §2.6(a) / §7).

Spec deviation (recorded in spec.md §"实施过程发现的方案缺陷"): the v2 CLI
named ``--dev dev_kcand``, but R6's "按 instance_id 的 candC 段聚合" requires
the per-candidate grouping that ``dev_kcand`` (built by ``build_alignment_data``,
out of this node's change scope) does not preserve under per-test trajectories.
We therefore read the per-test trajectory ``test`` split directly via
``--traj`` — equivalent, self-contained, and still MBPP-only.

Usage:
    python le-wm-JR/scripts/calibrate_consensus.py \\
        --wm-ckpt <stage0 best> --wm-config wm_humaneval \\
        --traj data/trajectories/humaneval_wm_v3_pertest \\
        --mbpp data/benchmarks/mbpp/mbpp.jsonl \\
        --search-theta 0.4,0.5,0.6 --search-mode soft,hard
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# le-wm-JR root + repo-root scripts on sys.path (for pca.* and _to_stub).
_LEWM_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LEWM_ROOT.parent
for _p in (str(_LEWM_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_alignment_data import _to_stub  # noqa: E402
from pca.inference.consensus import consensus_rank  # noqa: E402
from pca.inference.wm_reranker import (  # noqa: E402
    WMReranker,
    WMRerankerConfig,
)


def _floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _modes(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _temp_grid() -> list[float]:
    return [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_mbpp(path: str) -> dict[int, dict]:
    rows = _read_jsonl(Path(path))
    return {int(r["task_id"]): r for r in rows if "task_id" in r}


def _parse_iid(iid: str):
    """``mbpp-000123-cand2-t1`` → ``(123, 2, 1)`` or ``(None, None, None)``."""
    parts = iid.split("-")
    if len(parts) < 4 or not parts[1].isdigit():
        return None, None, None
    if not (parts[2].startswith("cand") and parts[3].startswith("t")):
        return None, None, None
    try:
        return int(parts[1]), int(parts[2][4:]), int(parts[3][1:])
    except ValueError:
        return None, None, None


def _parse_candidate(obs: str) -> str:
    """Recover the CANDIDATE program from a ``serialize_test`` observation."""
    mark = "\nCANDIDATE:\n"
    if mark not in obs:
        return ""
    return obs.split(mark, 1)[1].rsplit("\nACTION:", 1)[0]


def _group_traj(rows: list[dict]) -> dict:
    """{tid: {cand: {'program': str, 'labels': {t: int}}}} (R6 dedup)."""
    groups: dict = {}
    for tr in rows:
        tid, cand, t = _parse_iid(tr.get("instance_id", ""))
        if tid is None:
            continue
        steps = tr.get("steps") or []
        if not steps:
            continue
        s0 = steps[0]
        g = groups.setdefault(tid, {}).setdefault(
            cand, {"program": None, "labels": {}}
        )
        if not g["program"]:
            g["program"] = _parse_candidate(s0.get("obs_text", ""))
        lab = s0.get("label")
        g["labels"][t] = 1 if (lab is not None and float(lab) >= 0.5) else 0
    return groups


def _dev_tasks(groups: dict, mbpp: dict) -> list[dict]:
    """Per task: candidate programs + tests + per-cand label + label matrix."""
    tasks: list[dict] = []
    for tid, cands in groups.items():
        row = mbpp.get(tid)
        tests = (row or {}).get("test_list") or []
        if row is None or not row.get("code") or not tests:
            continue
        stub = _to_stub(row, row["code"])
        n_t = len(tests)
        programs, per_cand, matrix = [], [], []
        for _c, g in sorted(cands.items()):
            if not g["program"]:
                continue
            labels = [g["labels"].get(t, 0) for t in range(n_t)]
            programs.append(g["program"])
            matrix.append(labels)
            per_cand.append(1 if all(v == 1 for v in labels) else 0)
        if len(programs) < 2:
            continue
        tasks.append({
            "tid": tid, "stub": stub, "programs": programs, "tests": tests,
            "per_cand": per_cand, "label_matrix": matrix,
        })
    return tasks


def _sigmoid(x: float, temp: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / max(temp, 1e-3)))


def _logit_matrices(reranker, tasks: list[dict]) -> list[list]:
    """Cache the raw ``(K, T)`` logit matrix per dev task (one WM pass each)."""
    mats = []
    for task in tasks:
        lg = reranker.score_matrix(
            task["stub"], task["programs"], task["tests"], return_logits=True
        )
        mats.append(lg.tolist())
    return mats


def _nll(logit_mats: list, label_mats: list, temp: float) -> float:
    total, n = 0.0, 0
    for lm, ym in zip(logit_mats, label_mats):
        for lr, yr in zip(lm, ym):
            for x, y in zip(lr, yr):
                p = min(max(_sigmoid(x, temp), 1e-6), 1 - 1e-6)
                total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                n += 1
    return total / max(n, 1)


def _ece(logit_mats: list, label_mats: list, temp: float,
         n_bins: int = 10) -> float:
    probs, labels = [], []
    for lm, ym in zip(logit_mats, label_mats):
        for lr, yr in zip(lm, ym):
            for x, y in zip(lr, yr):
                probs.append(_sigmoid(x, temp))
                labels.append(y)
    if not probs:
        return 0.0
    total, n = 0.0, float(len(probs))
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(probs)
               if (p > lo or (i == 0 and p >= lo)) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[j] for j in idx) / len(idx)
        acc = sum(labels[j] for j in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def _fit_temp(logit_mats: list, label_mats: list):
    best_t, best_nll = 1.0, 1e18
    for t in _temp_grid():
        nll = _nll(logit_mats, label_mats, t)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t, _ece(logit_mats, label_mats, 1.0), \
        _ece(logit_mats, label_mats, best_t)


def _consensus_acc(logit_mats, tasks, temp, theta, mode) -> float:
    correct = 0
    for mat, task in zip(logit_mats, tasks):
        probs = [[_sigmoid(x, temp) for x in row] for row in mat]
        scores = consensus_rank(
            probs, [0.0] * len(probs), theta=theta, mode=mode
        )
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += task["per_cand"][pred]
    return correct / max(len(tasks), 1)


def _baseline_acc(logit_mats, tasks, temp) -> float:
    """Mean predicted-pass argmax (the plain global-verifier pick)."""
    correct = 0
    for mat, task in zip(logit_mats, tasks):
        means = [sum(_sigmoid(x, temp) for x in r) / len(r) for r in mat]
        pred = max(range(len(means)), key=lambda i: means[i])
        correct += task["per_cand"][pred]
    return correct / max(len(tasks), 1)


def _select(logit_mats, tasks, temp, args) -> dict:
    """Grid-search theta × mode under the non-destructive guardrail."""
    grid = {}
    for mode in _modes(args.search_mode):
        for theta in _floats(args.search_theta):
            grid[(mode, theta)] = _consensus_acc(
                logit_mats, tasks, temp, theta, mode
            )
    best = max(grid, key=lambda k: grid[k])
    base = _baseline_acc(logit_mats, tasks, temp)
    if grid[best] > base + 1e-9:
        mode, theta, guard = best[0], best[1], False
    else:
        mode, theta, guard = "soft", 0.5, True
    return {
        "consensus_mode": mode, "theta": theta, "w_l": 0.0,
        "dev": {"acc": round(grid[best], 4), "baseline_argmax": round(base, 4),
                "guardrail": guard, "n_dev": len(tasks),
                "grid": {f"{m}_{th}": round(v, 4)
                         for (m, th), v in grid.items()}},
    }


def _build_reranker(args) -> WMReranker:
    import torch

    cfg = WMRerankerConfig(
        wm_config_name=args.wm_config, ckpt_path=args.wm_ckpt,
        score_mode="verifier",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"[calib_consensus] WMReranker {args.wm_config} ckpt={args.wm_ckpt}")
    return WMReranker(cfg)


def run(args) -> int:
    mbpp = _load_mbpp(args.mbpp)
    rows = _read_jsonl(Path(args.traj) / "test.jsonl")
    tasks = _dev_tasks(_group_traj(rows), mbpp)
    if not tasks:
        raise SystemExit(
            f"[calib_consensus] no dev tasks from {args.traj}/test.jsonl"
        )
    reranker = _build_reranker(args)
    logit_mats = _logit_matrices(reranker, tasks)
    label_mats = [t["label_matrix"] for t in tasks]
    temp, ece_b, ece_a = _fit_temp(logit_mats, label_mats)
    chosen = _select(logit_mats, tasks, temp, args)
    print(f"[calib_consensus] T={temp:.3f} ECE {ece_b:.4f}->{ece_a:.4f} "
          f"mode={chosen['consensus_mode']} theta={chosen['theta']} "
          f"acc={chosen['dev']['acc']} base={chosen['dev']['baseline_argmax']} "
          f"guardrail={chosen['dev']['guardrail']} n_dev={len(tasks)}")

    out_dir = Path(args.out) if args.out else Path(args.wm_ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "verifier_temp": temp,
        "theta": chosen["theta"],
        "consensus_mode": chosen["consensus_mode"],
        "w_l": chosen["w_l"],
        "dev": chosen["dev"],
        "ece": {"before": round(ece_b, 4), "after": round(ece_a, 4)},
        "leak_check": "dev/MBPP only; HumanEval never read",
    }
    out_path = out_dir / "consensus_config.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib_consensus] wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", required=True, help="Stage-0 best ckpt")
    ap.add_argument("--wm-config", default="wm_humaneval")
    ap.add_argument("--traj", required=True,
                    help="per-test trajectory dir (reads its test.jsonl)")
    ap.add_argument("--mbpp", default="data/benchmarks/mbpp/mbpp.jsonl")
    ap.add_argument("--search-theta", default="0.4,0.5,0.6")
    ap.add_argument("--search-mode", default="soft,hard")
    ap.add_argument("--search-wl", default="0.0",
                    help="reserved; w_l fusion needs the aligned model (P1)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: alongside --wm-ckpt)")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
