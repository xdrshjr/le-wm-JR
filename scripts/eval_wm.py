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

# Allow ``python scripts/eval_wm.py`` from le-wm-JR root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _eval_outcome_auroc(model, loader, ctx_len: int, n_preds: int, device):
    """Val AUROC of P(pass) vs. binarized label (status==PASS); spec §6.2."""
    if getattr(model, "outcome_head", None) is None:
        return None
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
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
            scores.extend(torch.sigmoid(logit).float().cpu().tolist())
            labels.extend(
                1 if v >= 1.0 - 1e-6 else 0 for v in label.tolist()
            )
    return _auroc(scores, labels)


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
        auroc_msg = f" auroc={auroc}"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[eval_wm] split={split} tag={tag} "
        f"pred_loss_mean={pred_loss:.6f} "
        f"norm_mse={scale_inv['norm_mse']:.4f} "
        f"cosine={scale_inv['cosine']:.4f}{auroc_msg} -> {out}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run()
