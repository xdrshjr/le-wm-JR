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

import torch
import torch.nn.functional as F


def _outcome_enabled(cfg) -> bool:
    """True iff a verifier OutcomeHead BCE term is configured (spec §F1)."""
    loss_cfg = cfg.get("loss") if hasattr(cfg, "get") else None
    outcome = loss_cfg.get("outcome") if loss_cfg is not None else None
    return bool(outcome) and bool(outcome.get("enabled", False))


def _exec_enabled(cfg) -> bool:
    """True iff the round-8 ExecTraceHead loss is configured (spec §2.2)."""
    loss_cfg = cfg.get("loss") if hasattr(cfg, "get") else None
    exec_cfg = loss_cfg.get("exec") if loss_cfg is not None else None
    return bool(exec_cfg) and bool(exec_cfg.get("enabled", False))


def _info_nce(o_hat, z_out, tau):
    """Symmetric in-batch InfoNCE between predicted ô and true outputs."""
    a = F.normalize(o_hat.float(), dim=-1)
    b = F.normalize(z_out.float(), dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.t(), target)
    )


def _match_bce(head, o_hat, z_out):
    """Output-equality BCE: positives (ô_i, z_i)=1, rolled negatives =0.

    Trains ``match_logit`` as the output-equality discriminator the PEC
    matrix relies on (spec §2.2 step 4). Skipped (returns 0) for B<2.
    """
    b = o_hat.size(0)
    if b < 2:
        return o_hat.new_zeros(())
    pos = head.match_logit(o_hat, z_out)
    neg = head.match_logit(o_hat, z_out.roll(1, dims=0))
    logit = torch.cat([pos, neg])
    tgt = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return F.binary_cross_entropy_with_logits(logit, tgt)


def _exec_loss(model, pred_emb, tgt_emb, cfg):
    """ExecTraceHead loss = β·InfoNCE(ô, z_true) + μ·output-equality BCE.

    Self-contained from the trajectory's output observation (tgt_emb encodes
    ``obs_next = serialize_output(y)``); no expected value needed — match is
    trained on in-batch positive/negative output pairs (spec §2.2 C-5).
    """
    head = model.exec_head
    o_hat = head.predict_output(pred_emb[:, -1])      # (B, P)
    z_out = head.embed_output(tgt_emb[:, -1])         # (B, P)
    beta = float(cfg.loss.exec.get("beta", 1.0))
    mu = float(cfg.loss.exec.get("mu", 0.5))
    nce = _info_nce(o_hat, z_out, head.tau)
    bce = _match_bce(head, o_hat, z_out)
    return beta * nce + mu * bce, nce.detach(), bce.detach()


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

    # Δ6 (spec §2.2): round-8 ExecTraceHead loss. Fully short-circuited when
    # loss.exec is absent/disabled or the model has no exec_head, so every
    # pre-R8 forward (incl. the verifier path above) is byte-identical.
    if _exec_enabled(cfg) and getattr(self.model, "exec_head", None):
        exec_loss, nce, bce = _exec_loss(self.model, pred_emb, tgt_emb, cfg)
        output["exec_loss"] = exec_loss
        output["exec_nce_loss"] = nce
        output["exec_match_loss"] = bce
        output["loss"] = output["loss"] + exec_loss

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)  # Δ4
    return output
