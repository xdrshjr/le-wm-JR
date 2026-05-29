"""TextJEPA — subclass of ``jepa.JEPA`` that swaps the vision ``encode()``
for a tool-IO text + op path. ``jepa.py`` is left bit-identical (R02).
"""
from __future__ import annotations

import torch
from einops import rearrange

from jepa import JEPA
from pca.action.schema import ExecutableOp


class TextJEPA(JEPA):
    """World model over text trajectories.

    ``info`` dict contract (PCA flavour):
        in:
            obs_text:  list[list[str]] shape (B, T)
            op:        list[list[ExecutableOp]] shape (B, T)
        out (written):
            emb:       (B, T, embed_dim)
            act_emb:   (B, T, embed_dim)

    Optionally carries an ``outcome_head`` (``pca.outcome_head.OutcomeHead``)
    that maps a predicted next-state embedding to a pass/fail logit for the
    verifier reranker (spec §3.5/§4.2). It stays ``None`` for goal_dist /
    MVP configs, keeping the base JEPA contract unchanged (R02).
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        outcome_head=None,
    ):
        super().__init__(encoder, predictor, action_encoder, projector,
                         pred_proj)
        # Registered as a submodule when provided (so it lands in the
        # state_dict); a plain ``None`` attribute otherwise.
        self.outcome_head = outcome_head

    def predict_outcome(self, pred_emb):
        """Map a predicted embedding to a pass logit.

        ``pred_emb`` may be (B, D) or (B, T, D); returns the logit with the
        trailing singleton squeezed. Raises if no head is attached.
        """
        if self.outcome_head is None:
            raise RuntimeError(
                "TextJEPA.predict_outcome called but outcome_head is None "
                "(train with loss.outcome.enabled=true / a verifier config)"
            )
        return self.outcome_head(pred_emb).squeeze(-1)

    def encode(self, info: dict) -> dict:
        obs_text: list[list[str]] = info["obs_text"]
        ops: list[list[ExecutableOp]] = info["op"]

        b = len(obs_text)
        if b == 0:
            raise ValueError("TextJEPA.encode received empty batch")
        t = len(obs_text[0])

        flat_text: list[str] = [s for row in obs_text for s in row]
        flat_ops: list[ExecutableOp] = [op for row in ops for op in row]

        encoded = self.encoder(flat_text)  # (B*T, base_out)
        emb_flat = self.projector(encoded)  # (B*T, embed_dim)
        info["emb"] = rearrange(emb_flat, "(b t) d -> b t d", b=b, t=t)

        act_flat = self.action_encoder(flat_ops)  # (B*T, embed_dim)
        info["act_emb"] = rearrange(
            act_flat, "(b t) d -> b t d", b=b, t=t
        )
        return info

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ):
        """Text-domain rollout is implemented inside PCAAgent (v2);
        the vision-style ``JEPA.rollout`` signature does not apply.
        """
        raise NotImplementedError(
            "TextJEPA.rollout is provided by PCAAgent in adaptation-spec v2"
        )

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        raise NotImplementedError(
            "TextJEPA.get_cost is provided by PCAAgent in adaptation-spec v2"
        )
