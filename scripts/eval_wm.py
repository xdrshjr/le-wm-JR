"""Compute val pred_loss for a TextJEPA, with optional checkpoint loading.

Two invocation modes:

  baseline (no ckpt, fresh random init):
      python scripts/eval_wm.py --config-name=wm_stage0_mvp split=val

  trained:
      python scripts/eval_wm.py --config-name=wm_stage0_mvp split=val \\
          ckpt=/path/to/weights_epoch_6.pt

Writes ``runs/pca_mvp/eval_<split>_<tag>.json`` and prints a short
summary that downstream scripts can grep.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

# Allow ``python scripts/eval_wm.py`` from le-wm-JR root; the scripts dir is
# also added so the shared ``diagnose_prediction.error_rates`` (D3) imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pca.data.collate import collate_trajectories  # noqa: E402
from pca.data.dataset import (  # noqa: E402
    TrajectoryDataset,
    TrajectoryDatasetConfig,
)


def _build_loader(cfg, split: str) -> DataLoader:
    ds_cfg = TrajectoryDatasetConfig(
        path=cfg.data.path,
        split=split,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        seed=cfg.seed,
        max_obs_chars=cfg.data.get("max_obs_chars", 16000),
    )
    ds = TrajectoryDataset(ds_cfg)
    return DataLoader(
        ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_trajectories,
        num_workers=0,
    )


def _eval_pred_loss(model, loader, ctx_len: int, n_preds: int, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            info = model.encode(batch)
            emb = info["emb"].to(device)
            act_emb = info["act_emb"].to(device)
            ctx_emb = emb[:, :ctx_len]
            ctx_act = act_emb[:, :ctx_len]
            tgt_emb = emb[:, n_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)
            mse = (pred_emb - tgt_emb).pow(2).mean().item()
            total += mse * emb.size(0)
            count += emb.size(0)
    return total / max(count, 1)


def _eval_scale_invariant(model, loader, ctx_len: int, n_preds: int, device):
    """Scale-invariant prediction quality (spec §6.2, dev-log R2).

    Raw ``pred_loss`` (MSE between predicted and target latents) is
    confounded for a SIGReg-regularised JEPA: SIGReg inflates the target
    embedding variance during training, so the absolute MSE can *rise*
    even as the prediction improves (and a *lower* raw MSE can merely mean
    representation collapse). These two metrics are scale-robust and are
    what the gate compares across the random-init baseline vs. trained:

      norm_mse = E[||pred - tgt||^2] / E[Var_dim(tgt)]   (lower = better)
      cosine   = mean cosine(pred, tgt)                  (higher = better)
    """
    import torch.nn.functional as F

    model.eval()
    sum_mse = sum_var = sum_cos = count = 0.0
    with torch.no_grad():
        for batch in loader:
            info = model.encode(batch)
            emb = info["emb"].to(device)
            act_emb = info["act_emb"].to(device)
            tgt = emb[:, n_preds:]
            pred = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
            p = pred.flatten(0, 1)
            t = tgt.flatten(0, 1)
            sum_mse += (p - t).pow(2).mean(-1).sum().item()
            sum_var += t.var(dim=-1, unbiased=False).sum().item()
            sum_cos += F.cosine_similarity(p, t, dim=-1).sum().item()
            count += p.size(0)
    count = max(count, 1)
    return {
        "norm_mse": sum_mse / max(sum_var, 1e-9),
        "cosine": sum_cos / count,
        "tgt_var": sum_var / count,
    }


def _auroc(scores: list[float], labels: list[int]) -> "float | None":
    """Mann-Whitney rank AUROC with tie handling. ``None`` if one-class."""
    import numpy as np

    if not scores:
        return None
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i, n = 0, len(s)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # avg of 1-based ranks
        i = j
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _outcome_enabled(cfg) -> bool:
    loss_cfg = cfg.get("loss", None)
    outcome = loss_cfg.get("outcome") if loss_cfg is not None else None
    return bool(outcome) and bool(outcome.get("enabled", False))


def _outcome_scores_labels(model, loader, ctx_len: int, device, temp=1.0):
    """Collect calibrated P(pass)=sigmoid(logit/T) + binarized labels."""
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    t = max(float(temp), 1e-3)
    with torch.no_grad():
        for batch in loader:
            label = batch.get("label")
            if label is None:
                continue
            info = model.encode(batch)
            emb = info["emb"].to(device)
            act_emb = info["act_emb"].to(device)
            pred_emb = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
            logit = model.outcome_head(pred_emb[:, -1]).squeeze(-1)
            scores.extend(torch.sigmoid(logit / t).float().cpu().tolist())
            labels.extend(
                1 if v >= 1.0 - 1e-6 else 0 for v in label.tolist()
            )
    return scores, labels


def _eval_outcome_auroc(model, loader, ctx_len: int, n_preds: int, device):
    """Val AUROC of P(pass) vs. binarized label (status==PASS); spec §6.2.

    On a per-test (``humaneval_wm_v3_pertest``) dataset every label is the
    binary "did this candidate pass this single assert" target, so this is
    the **per-test** AUROC the PEC gate (>=0.75) is selected on (spec §2.3).
    On the legacy ratio-label data it is the global pass AUROC, unchanged.
    """
    if getattr(model, "outcome_head", None) is None:
        return None
    scores, labels = _outcome_scores_labels(model, loader, ctx_len, device)
    return _auroc(scores, labels)


def _ece(scores: list[float], labels: list[int], n_bins: int = 10):
    """Expected Calibration Error (reliability); ``None`` if empty (P1)."""
    import numpy as np

    if not scores:
        return None
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, float(len(s))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (s > lo) & (s <= hi) if i > 0 else (s >= lo) & (s <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(float(y[m].mean()) - float(s[m].mean()))
    return float(total)


# ----- Stage-1 alignment quality (wm-llm-alignment §6, P1) -------------


def _read_pairs(pairs_path: str) -> list[dict]:
    return [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _wm_latents(model, obs_texts: list[str], device, cap: int = 4000):
    """Predicted ẑ₁ for each step0 observation → (N, d_wm)."""
    from pca.action.schema import RunTestArgs

    op = RunTestArgs(selector="visible_tests", timeout_sec=5)
    info = {
        "obs_text": [[t[:cap]] for t in obs_texts],
        "op": [[op] for _ in obs_texts],
    }
    with torch.no_grad():
        info = model.encode(info)
        z1 = model.predict(info["emb"][:, :1], info["act_emb"][:, :1])[:, -1]
    return z1.float().to(device)


def _text_vectors(llm_name: str, texts: list[str], device):
    """Frozen LLM masked-mean sentence vectors → ((N, d_llm), d_llm)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModel.from_pretrained(
        llm_name, torch_dtype=torch.float16
    ).to(device).eval()
    enc = tok(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = lm(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return pooled.float(), int(lm.config.hidden_size)


def _projector_from_ckpt(ckpt: str, d_llm: int):
    from pca.projector.mlp import (
        WorldModelProjector,
        WorldModelProjectorConfig,
    )

    sd = torch.load(ckpt, map_location="cpu")
    k = max(1, int(sd["fc2.weight"].shape[0]) // d_llm)
    pcfg = WorldModelProjectorConfig(
        in_dim=int(sd["fc1.weight"].shape[1]),
        hidden_dim=int(sd["fc1.weight"].shape[0]),
        out_dim=d_llm,
        num_tokens=k,
        dtype=torch.float32,
    )
    proj = WorldModelProjector(pcfg)
    proj.load_state_dict(sd, strict=False)
    return proj


def _retrieval_at_1(p, t) -> float:
    import torch.nn.functional as F

    p = F.normalize(p.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    pred = (p @ t.t()).argmax(dim=-1)
    target = torch.arange(p.size(0), device=p.device)
    return float((pred == target).float().mean().item())


def _eval_alignment_quality(cfg, model, device):
    """Retrieval@1 of projected ẑ₁ → result-text vector (spec §6, P1).

    Pure-additive: returns ``None`` unless ``align_ckpt`` + ``pairs`` are
    set, so the standard WM eval path is unchanged.
    """
    align_ckpt = cfg.get("align_ckpt")
    pairs_path = cfg.get("pairs")
    if not align_ckpt or not pairs_path:
        return None
    rows = _read_pairs(pairs_path)[: int(cfg.get("align_limit", 256))]
    if not rows:
        return None
    llm_name = cfg.get("llm_name", "Qwen/Qwen2.5-1.5B-Instruct")
    t, d_llm = _text_vectors(
        llm_name, [r["outcome_text"] for r in rows], device
    )
    proj = _projector_from_ckpt(align_ckpt, d_llm).to(device).eval()
    z1 = _wm_latents(model, [r["step0_obs"] for r in rows], device)
    with torch.no_grad():
        p = proj(z1).mean(dim=1)
    return {"retrieval_at_1": _retrieval_at_1(p, t), "n_pairs": len(rows)}


@hydra.main(
    version_base=None,
    config_path="../config/train",
    config_name="wm_stage0_mvp",
)
def run(cfg):
    split = cfg.get("split", "val")
    ckpt = cfg.get("ckpt", None)
    tag = cfg.get("tag", "trained" if ckpt else "baseline_random_init")

    model = hydra.utils.instantiate(cfg.model)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[eval_wm] loaded {ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print("[eval_wm] baseline mode — random init, no checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = _build_loader(cfg, split)
    pred_loss = _eval_pred_loss(
        model,
        loader,
        ctx_len=cfg.wm.history_size,
        n_preds=cfg.wm.num_preds,
        device=device,
    )

    scale_inv = _eval_scale_invariant(
        model, loader,
        ctx_len=cfg.wm.history_size,
        n_preds=cfg.wm.num_preds,
        device=device,
    )

    out_dir = Path(cfg.get("output_dir", "runs/pca_mvp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{split}_{tag}.json"
    payload = {
        "split": split,
        "tag": tag,
        "ckpt": ckpt,
        "pred_loss_mean": pred_loss,
        # Scale-invariant gate metrics (robust to SIGReg variance growth;
        # compare these across baseline vs. trained, not raw pred_loss).
        "norm_mse": scale_inv["norm_mse"],
        "cosine": scale_inv["cosine"],
        "tgt_var": scale_inv["tgt_var"],
        "data_path": cfg.data.path,
        "embed_dim": cfg.wm.embed_dim,
        "history_size": cfg.wm.history_size,
        "num_preds": cfg.wm.num_preds,
    }
    # Verifier AUROC is a pure-additive field — only present when the
    # outcome head is enabled, so goal_dist / MVP output is unchanged
    # (spec §F2, §6.4).
    auroc_msg = ""
    if _outcome_enabled(cfg):
        auroc = _eval_outcome_auroc(
            model, loader,
            ctx_len=cfg.wm.history_size,
            n_preds=cfg.wm.num_preds,
            device=device,
        )
        payload["auroc"] = auroc
        # Calibrated reliability (spec §3 P1): ECE at the configured verifier
        # temperature (1.0 = uncalibrated). Pure-additive fields.
        temp = float(cfg.get("verifier_temp", 1.0))
        sc, lab = _outcome_scores_labels(
            model, loader, cfg.wm.history_size, device, temp
        )
        payload["ece"] = _ece(sc, lab)
        payload["verifier_temp"] = temp
        # R7 (spec §3 / D3): reuse the same flat (prob,label) aggregation as
        # diagnose_prediction — Brier + false-pos/neg rates. Pure-additive; the
        # candidate/test-side decomposition (needs the (K,T) structure) lives
        # in diagnose_prediction.py. Best-effort: skip if import unavailable.
        try:
            from diagnose_prediction import error_rates

            payload.update(error_rates(sc, lab))
        except Exception:
            pass
        auroc_msg = f" auroc={auroc} ece={payload['ece']}"

    # Stage-1 alignment quality — additive, only when align_ckpt+pairs set.
    align = _eval_alignment_quality(cfg, model, device)
    align_msg = ""
    if align is not None:
        payload["alignment"] = align
        align_msg = f" align@1={align['retrieval_at_1']:.3f}"

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[eval_wm] split={split} tag={tag} "
        f"pred_loss_mean={pred_loss:.6f} "
        f"norm_mse={scale_inv['norm_mse']:.4f} "
        f"cosine={scale_inv['cosine']:.4f}{auroc_msg}{align_msg} -> {out}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run()
