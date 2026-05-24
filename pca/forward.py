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

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)  # Δ4
    return output
