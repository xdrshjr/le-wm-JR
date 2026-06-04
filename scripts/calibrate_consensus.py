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
        --traj data/trajectories/humaneval_wm_v4_pertest_hardneg \\
        --mbpp data/benchmarks/mbpp/mbpp.jsonl \\
        --search-theta 0.4,0.5,0.6 --search-mode soft,soft_conf,hard \\
        --search-gamma 0.5,1.0,2.0
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


def _consensus_acc(logit_mats, tasks, temp, cfg) -> float:
    """Dev rerank accuracy for one (theta, mode, gamma) point.

    ``cfg`` = ``(theta, mode, gamma)``; gamma only affects ``soft_conf``
    (spec §2.4), so soft/hard grids stay identical to round 6.
    """
    theta, mode, gamma = cfg
    correct = 0
    for mat, task in zip(logit_mats, tasks):
        probs = [[_sigmoid(x, temp) for x in row] for row in mat]
        scores = consensus_rank(
            probs, [0.0] * len(probs), theta=theta, mode=mode, gamma=gamma
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
    """Grid-search theta × mode × gamma under the non-destructive guardrail.

    Gamma is only swept for ``soft_conf`` (it is a no-op elsewhere); other
    modes are pinned to gamma=1.0 so the grid does not duplicate them.
    """
    gammas = _floats(args.search_gamma)
    grid = {}
    for mode in _modes(args.search_mode):
        mode_gammas = gammas if mode == "soft_conf" else [1.0]
        for theta in _floats(args.search_theta):
            for gamma in mode_gammas:
                grid[(mode, theta, gamma)] = _consensus_acc(
                    logit_mats, tasks, temp, (theta, mode, gamma)
                )
    best = max(grid, key=lambda k: grid[k])
    base = _baseline_acc(logit_mats, tasks, temp)
    if grid[best] > base + 1e-9:
        mode, theta, gamma, guard = best[0], best[1], best[2], False
    else:
        mode, theta, gamma, guard = "soft", 0.5, 1.0, True
    return {
        "consensus_mode": mode, "theta": theta, "gamma": gamma, "w_l": 0.0,
        "dev": {"acc": round(grid[best], 4), "baseline_argmax": round(base, 4),
                "guardrail": guard, "n_dev": len(tasks),
                "grid": {f"{m}_{th}_g{g}": round(v, 4)
                         for (m, th, g), v in grid.items()}},
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


# ----- round-8 exec-mode calibration (spec §2.4.1 / §2.6) --------------


def _is_exec(args) -> bool:
    return bool(getattr(args, "exec_mode", False)) or "exec" in args.wm_config


def _parse_exec_obs(obs: str) -> tuple[str, str, str]:
    """serialize_exec obs → ``(problem, input, program)`` (best-effort)."""
    problem = inp = prog = ""
    if "\nINPUT:\n" in obs:
        problem = obs.split("PROBLEM:\n", 1)[-1].split("\nINPUT:\n", 1)[0]
        rest = obs.split("\nINPUT:\n", 1)[1]
        inp = rest.split("\nCANDIDATE:\n", 1)[0]
    if "\nCANDIDATE:\n" in obs:
        prog = obs.split("\nCANDIDATE:\n", 1)[1].rsplit("\nACTION:", 1)[0]
    return problem, inp, prog


def _group_exec(rows: list[dict]) -> dict:
    """{key: {cand: {program, stub, inputs[], expected[], labels[]}}}."""
    groups: dict = {}
    for tr in rows:
        iid = tr.get("instance_id", "")
        steps = tr.get("steps") or []
        if "-cand" not in iid or not steps:
            continue
        s0 = steps[0]
        key, cseg = iid.rsplit("-cand", 1)
        try:
            cand = int(cseg.split("-")[0])
        except ValueError:
            continue
        problem, inp, prog = _parse_exec_obs(s0.get("obs_text", ""))
        g = groups.setdefault(key, {}).setdefault(
            cand, {"program": prog, "stub": problem,
                   "inputs": [], "expected": [], "labels": []}
        )
        g["inputs"].append(inp)
        g["expected"].append(s0.get("expected"))
        g["labels"].append(s0.get("label"))
    return groups


def _exec_dev_tasks(groups: dict) -> list[dict]:
    """Per problem: candidate programs + shared inputs/expected + labels."""
    tasks: list[dict] = []
    for cands in groups.values():
        ref = max(cands.values(), key=lambda g: len(g["inputs"]))
        programs, per_cand = [], []
        for _c, g in sorted(cands.items()):
            if not g["program"]:
                continue
            labels = [v for v in g["labels"] if v is not None]
            programs.append(g["program"])
            per_cand.append(1 if labels and all(v >= 0.5 for v in labels)
                            else 0)
        if len(programs) < 2:
            continue
        tasks.append({
            "stub": ref["stub"], "programs": programs,
            "inputs": ref["inputs"], "expected": ref["expected"],
            "per_cand": per_cand,
        })
    return tasks


def _exec_cache(reranker, tasks: list[dict]) -> list:
    """Cache (ô, z_exp) per task — one WM pass each (spec §2.4.1)."""
    cache = []
    for task in tasks:
        has_exp = all(e is not None for e in task["expected"])
        o_hat, z_exp = reranker.exec_embeddings(
            task["stub"], task["programs"], task["inputs"],
            task["expected"] if has_exp else None,
        )
        cache.append((o_hat, z_exp))
    return cache


def _exec_acc(cache, tasks, tau, cluster_thr, cfg_mgt) -> float:
    """Dev rerank acc for one (τ, cluster_thr, mode, gamma, theta)."""
    from pca.inference.consensus import (
        consensus_rank,
        exec_pass_from_outputs,
    )

    mode, gamma, theta = cfg_mgt
    correct = 0
    for (o_hat, z_exp), task in zip(cache, tasks):
        mat = exec_pass_from_outputs(
            o_hat, z_exp, tau=tau, cluster_thr=cluster_thr
        )
        scores = consensus_rank(mat, [0.0] * len(mat), theta=theta,
                                mode=mode, gamma=gamma)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += task["per_cand"][pred]
    return correct / max(len(tasks), 1)


def _select_exec(cache, tasks, args) -> dict:
    """Grid τ × cluster_thr × mode × gamma × theta, guardrailed (spec §2.6)."""
    best, best_acc = None, -1.0
    for mode in _modes(args.search_mode):
        gammas = _floats(args.search_gamma) if mode == "soft_conf" else [1.0]
        for theta in _floats(args.search_theta):
            for gamma in gammas:
                for tau in _floats(args.search_tau):
                    for cthr in _floats(args.search_cluster_thr):
                        acc = _exec_acc(cache, tasks, tau, cthr,
                                        (mode, gamma, theta))
                        if acc > best_acc:
                            best_acc, best = acc, (mode, theta, gamma,
                                                   tau, cthr)
    base = _exec_baseline(cache, tasks)
    guard = best_acc <= base + 1e-9
    if guard:
        best = ("soft", 0.5, 1.0, 1.0, 0.7)
    mode, theta, gamma, tau, cthr = best
    return {
        "consensus_mode": mode, "theta": theta, "gamma": gamma, "w_l": 0.0,
        "exec": {"tau": tau, "cluster_thr": cthr,
                 "use_agreement_for_no_doctest": True},
        "dev": {"acc": round(best_acc, 4), "baseline": round(base, 4),
                "guardrail": guard, "n_dev": len(tasks)},
    }


def _exec_baseline(cache, tasks) -> float:
    """Mean predicted output-match argmax (no consensus): the guardrail base."""
    from pca.inference.consensus import exec_pass_from_outputs

    correct = 0
    for (o_hat, z_exp), task in zip(cache, tasks):
        mat = exec_pass_from_outputs(o_hat, z_exp, tau=1.0, cluster_thr=0.7)
        means = [sum(r) / max(len(r), 1) for r in mat]
        pred = max(range(len(means)), key=lambda i: means[i])
        correct += task["per_cand"][pred]
    return correct / max(len(tasks), 1)


def _run_exec(args) -> int:
    import torch

    rows = _read_jsonl(Path(args.traj) / "test.jsonl")
    tasks = _exec_dev_tasks(_group_exec(rows))
    if not tasks:
        raise SystemExit(
            f"[calib_consensus] no exec dev tasks from {args.traj}/test.jsonl"
        )
    cfg = WMRerankerConfig(
        wm_config_name=args.wm_config, ckpt_path=args.wm_ckpt,
        score_mode="exec",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    reranker = WMReranker(cfg)
    chosen = _select_exec(_exec_cache(reranker, tasks), tasks, args)
    print(f"[calib_consensus] EXEC mode={chosen['consensus_mode']} "
          f"theta={chosen['theta']} gamma={chosen['gamma']} "
          f"tau={chosen['exec']['tau']} cthr={chosen['exec']['cluster_thr']} "
          f"acc={chosen['dev']['acc']} base={chosen['dev']['baseline']} "
          f"guardrail={chosen['dev']['guardrail']} n_dev={len(tasks)}")
    out_dir = Path(args.out) if args.out \
        else Path(args.wm_ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "verifier_temp": 1.0,
        "theta": chosen["theta"],
        "consensus_mode": chosen["consensus_mode"],
        "gamma": chosen["gamma"],
        "w_l": chosen["w_l"],
        "exec": chosen["exec"],
        "dev": chosen["dev"],
        "leak_check": "dev/MBPP/multi-src only; HumanEval never read",
    }
    out_path = out_dir / "consensus_config.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib_consensus] wrote {out_path}")
    return 0


def run(args) -> int:
    if _is_exec(args):
        return _run_exec(args)
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
          f"gamma={chosen['gamma']} "
          f"acc={chosen['dev']['acc']} base={chosen['dev']['baseline_argmax']} "
          f"guardrail={chosen['dev']['guardrail']} n_dev={len(tasks)}")

    out_dir = Path(args.out) if args.out else Path(args.wm_ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "verifier_temp": temp,
        "theta": chosen["theta"],
        "consensus_mode": chosen["consensus_mode"],
        "gamma": chosen["gamma"],
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
    ap.add_argument("--search-mode", default="soft,soft_conf,hard",
                    help="CodeT modes to grid (R7 adds soft_conf; spec §2.4)")
    ap.add_argument("--search-gamma", default="0.5,1.0,2.0",
                    help="confidence exponent grid for soft_conf (spec §2.4)")
    ap.add_argument("--search-wl", default="0.0",
                    help="reserved; w_l fusion needs the aligned model (P1)")
    # R8 exec-mode grids (spec §2.4.1): output-similarity τ + cluster threshold.
    ap.add_argument("--exec", dest="exec_mode", action="store_true",
                    help="calibrate the exec PEC matrix (auto-on if wm-config "
                         "name contains 'exec'); writes the exec block")
    ap.add_argument("--search-tau", default="0.5,1.0,2.0",
                    help="exec output-similarity temperature grid (§2.4.1)")
    ap.add_argument("--search-cluster-thr", default="0.6,0.75",
                    help="exec consistency cluster-threshold grid (§2.4.1)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: alongside --wm-ckpt)")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
