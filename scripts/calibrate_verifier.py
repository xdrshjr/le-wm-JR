"""calibrate_verifier — fit the verifier temperature + fusion weights.

Two independent, leakage-free calibrations (spec §2.5 step 4 / P0-2):

  (a) temperature ``T`` — on the instruct val split (single draft +
      ``exec_label``), minimise NLL of ``sigmoid(logit/T)``; report ECE
      before/after. ``T`` calibrates the per-sample probability
      (reliability), not cross-term scale (spec §2.3.3 / P1-A).
  (b) ``alpha_pos`` (+ optional ``w_t``) — on a held-out K-candidate
      MBPP-val dev set (``dev_kcand/dev.jsonl``, the trajectory ``test``
      split, never trained on by Stage-0/Stage-2 and never HumanEval).
      Grid-search fused accuracy and enforce the non-destructive guardrail:
      if the best fused config does not strictly beat verifier-only, fall
      back to ``alpha_pos=1.0`` (pure verifier), which is mathematically ≥
      verifier-only (spec §2.2 / §7).

Writes the chosen values into ``<aligned_ckpt>/align_config.json``
(``verifier_temp``, ``alpha_pos``, ``w_t``, plus an ``ece``/``dev`` audit
block). **Physically reads only val/dev — never HumanEval (spec §6 red
line).**

Usage:
    python le-wm-JR/scripts/calibrate_verifier.py \\
        --aligned-ckpt runs/pca_align/<job>/stage2 \\
        --wm-ckpt <stage0 best> --wm-config wm_humaneval \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --val data/align/v2 --dev data/align/v2/dev_kcand \\
        --search-alpha 0.7,0.85,1.0 --search-wt 0.0,0.3,0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# le-wm-JR root on sys.path so ``pca.*`` resolves on direct execution.
_LEWM_ROOT = Path(__file__).resolve().parents[1]
if str(_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEWM_ROOT))

from pca.aligned_model import AlignedWMLLM  # noqa: E402
from pca.training.llava_stages import InstructDataset  # noqa: E402


def _floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _temp_grid() -> list[float]:
    """Log-ish temperature grid for the NLL minimiser (spec §2.3.3)."""
    base = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
    return base


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width bins."""
    if probs.numel() == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, float(probs.numel())
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if i == 0:
            mask = mask | (probs == lo)
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        conf = float(probs[mask].mean().item())
        acc = float(labels[mask].mean().item())
        total += (cnt / n) * abs(acc - conf)
    return total


def _collect_val_logits(model, rows: list[dict]):
    """Raw verifier logits + binarised labels over the instruct val split."""
    logits: list[float] = []
    labels: list[float] = []
    for r in rows:
        lg = model.verifier_logits(r["problem"], [r.get("draft", "")])
        logits.append(float(lg[0]))
        lab = 1.0 if float(r.get("exec_label", 0.0)) >= 1.0 - 1e-6 else 0.0
        labels.append(lab)
    return torch.tensor(logits), torch.tensor(labels)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor):
    """Pick ``T`` minimising val NLL; return ``(T, ece_before, ece_after)``."""
    if logits.numel() == 0:
        return 1.0, 0.0, 0.0
    ece_before = _ece(torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6), labels)
    best_t, best_nll = 1.0, 1e18
    for t in _temp_grid():
        p = torch.sigmoid(logits / t).clamp(1e-6, 1 - 1e-6)
        nll = -(labels * torch.log(p)
                + (1 - labels) * torch.log(1 - p)).mean().item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    p_cal = torch.sigmoid(logits / best_t).clamp(1e-6, 1 - 1e-6)
    return best_t, ece_before, _ece(p_cal, labels)


def _grid_acc(model, cache: list, alpha: float, wt: float) -> float:
    """Fused argmax accuracy over the cached dev terms at ``(alpha, wt)``."""
    model.alpha_pos = alpha
    model.w_self_test = wt
    correct, total = 0, 0
    for terms, labels, n_vis in cache:
        score = model._fuse_terms(terms, n_vis)
        pred = int(torch.argmax(score).item())
        correct += int(labels[pred])
        total += 1
    return correct / max(total, 1)


def _cache_dev_terms(model, rows: list[dict], want_self: bool) -> list:
    """Pre-compute raw signal vectors once per dev task (spec §2.2 cache)."""
    cache = []
    for r in rows:
        terms = model._score_terms(
            r["problem"], r["programs"], r["completions"],
            force_self_test=want_self,
        )
        cache.append((terms, r["labels"], int(r.get("n_visible", 1))))
    return cache


def _select(results: dict, vonly: float):
    """Best fused config under the non-destructive guardrail (spec §2.2)."""
    (alpha, wt) = max(results, key=lambda k: results[k])
    acc = results[(alpha, wt)]
    if acc > vonly + 1e-9 and alpha < 1.0:
        return {"alpha_pos": alpha, "w_t": wt, "fused_acc": acc,
                "verifier_acc": vonly, "guardrail": False}
    return {"alpha_pos": 1.0, "w_t": 0.0, "fused_acc": acc,
            "verifier_acc": vonly, "guardrail": True}


def _calibrate_alpha(model, dev_rows: list[dict], args) -> dict:
    """Grid-search ``alpha_pos`` / ``w_t`` on the held-out dev set."""
    alphas, wts = _floats(args.search_alpha), _floats(args.search_wt)
    want_self = any(w > 0 for w in wts)
    if want_self:
        _attach_proposer(model)
    cache = _cache_dev_terms(model, dev_rows, want_self)
    results = {(a, w): _grid_acc(model, cache, a, w)
               for a in alphas for w in wts}
    vonly = _grid_acc(model, cache, 1.0, 0.0)
    chosen = _select(results, vonly)
    chosen["n_dev"] = len(dev_rows)
    chosen["grid"] = {f"a{a}_w{w}": v for (a, w), v in results.items()}
    return chosen


def _attach_proposer(model) -> None:
    from pca.inference.test_proposer import TestProposer

    model.test_proposer = TestProposer(model.tokenizer, model.llm)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_config(aligned_ckpt: str, payload: dict) -> None:
    d = Path(aligned_ckpt)
    cfg_path = d / "align_config.json"
    meta = {}
    if cfg_path.exists():
        meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    meta.update(payload)
    cfg_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {cfg_path}")


def _load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calibrate] loading AlignedWMLLM wm={args.wm_config} "
          f"ckpt={args.wm_ckpt} aligned={args.aligned_ckpt} ({device})")
    return AlignedWMLLM(
        wm_cfg_name=args.wm_config,
        wm_ckpt=args.wm_ckpt,
        llm_name=args.model,
        aligned_ckpt=args.aligned_ckpt,
        device=device,
    )


def run(args) -> int:
    model = _load_model(args)
    val_rows = InstructDataset(args.val, "val", 0.1, 0).rows
    logits, labels = _collect_val_logits(model, val_rows)
    temp, ece_b, ece_a = _fit_temperature(logits, labels)
    model.verifier_temp = temp
    print(f"[calibrate] T={temp:.3f} ECE {ece_b:.4f}->{ece_a:.4f} "
          f"(n_val={len(val_rows)})")

    dev_rows = _read_jsonl(Path(args.dev) / "dev.jsonl")
    if dev_rows:
        chosen = _calibrate_alpha(model, dev_rows, args)
        print(f"[calibrate] alpha_pos={chosen['alpha_pos']} "
              f"w_t={chosen['w_t']} fused={chosen['fused_acc']:.4f} "
              f"verifier={chosen['verifier_acc']:.4f} "
              f"guardrail={chosen['guardrail']}")
    else:
        print("[calibrate] WARN: no dev set — keeping default alpha_pos/w_t")
        chosen = {"alpha_pos": model.alpha_pos, "w_t": model.w_self_test,
                  "n_dev": 0}

    payload = {
        "verifier_temp": temp,
        "alpha_pos": chosen["alpha_pos"],
        "w_t": chosen["w_t"],
        "ece": {"before": ece_b, "after": ece_a, "n_val": len(val_rows)},
        "dev": chosen,
        "leak_check": "val/dev only; HumanEval never read",
    }
    _write_config(args.aligned_ckpt, payload)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-ckpt", required=True,
                    help="Stage-2 product dir (align_config.json written here)")
    ap.add_argument("--wm-ckpt", required=True, help="Stage-0 best ckpt")
    ap.add_argument("--wm-config", default="wm_humaneval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--val", required=True,
                    help="instruct dir (val split) for temperature only")
    ap.add_argument("--dev", required=True,
                    help="dev_kcand dir for alpha_pos + guardrail")
    ap.add_argument("--search-alpha", default="0.7,0.85,1.0")
    ap.add_argument("--search-wt", default="0.0,0.3,0.5")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
