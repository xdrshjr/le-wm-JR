"""pca_forward — text-trajectory analogue of ``train.lejepa_forward`` (R09).

4-line delta from upstream:
    Δ1  removed ``torch.nan_to_num(batch["action"], 0.0)``
        — ``ExecutableOp`` is a dict, not a numeric tensor.
    Δ2  ``self.model.encode(batch)`` — same call, but ``self.model`` is
        a ``TextJEPA`` whose ``encode()`` consumes
        ``batch["obs_text"]`` + ``batch["op"]`` and writes
        ``info["emb"]`` + ``info["act_emb"]`` with identical shapes.
    Δ3  ``SIGReg`` is shape-agnostic over ``D`` — unchanged.
    Δ4  ``losses_dict`` key prefix ``{stage}/...`` — unchanged.
"""
from __future__ import annotations

import torch.nn.functional as F


def _outcome_enabled(cfg) -> bool:
    """True iff a verifier OutcomeHead BCE term is configured (spec §F1)."""
    loss_cfg = cfg.get("loss") if hasattr(cfg, "get") else None
    outcome = loss_cfg.get("outcome") if loss_cfg is not None else None
    return bool(outcome) and bool(outcome.get("enabled", False))


def pca_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Δ1: no nan_to_num for non-tensor op dicts.

    output = self.model.encode(batch)  # Δ2: TextJEPA.encode

    emb = output["emb"]          # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, D)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))  # Δ3
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Δ5 (spec §4.2 / F1): optional verifier head. Fully short-circuited
    # when loss.outcome is absent/disabled or the batch carries no label —
    # the path above is then byte-identical to the MVP forward.
    label = batch.get("label")
    if _outcome_enabled(cfg) and label is not None:
        mu = cfg.loss.outcome.get("weight", 1.0)
        logit = self.model.outcome_head(pred_emb[:, -1]).squeeze(-1)  # (B,)
        target = label.to(logit.device, dtype=logit.dtype)
        output["outcome_loss"] = F.binary_cross_entropy_with_logits(
            logit, target
        )
        output["loss"] = output["loss"] + mu * output["outcome_loss"]

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)  # Δ4
    return output
