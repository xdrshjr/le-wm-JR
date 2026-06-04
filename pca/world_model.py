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

    Round-8 (spec wm-exec-trace-fusion-sota §2.2) additionally allows an
    ``exec_head`` (``pca.exec_head.ExecTraceHead``) that predicts the output
    *embedding* a candidate produces and scores output equality. Like
    ``outcome_head`` it stays ``None`` for every pre-R8 config, so the base
    contract is byte-identical when no ``exec_head`` is configured.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        outcome_head=None,
        exec_head=None,
    ):
        super().__init__(encoder, predictor, action_encoder, projector,
                         pred_proj)
        # Registered as a submodule when provided (so it lands in the
        # state_dict); a plain ``None`` attribute otherwise.
        self.outcome_head = outcome_head
        self.exec_head = exec_head

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

    def predict_output(self, pred_emb):
        """Map a predicted latent ẑ₁ to its output embedding ô (spec §2.2).

        ``pred_emb`` may be (B, D) or (B, T, D); returns (..., proj_dim).
        Raises if no ``exec_head`` is attached, mirroring ``predict_outcome``
        — but the short-circuit in ``pca_forward`` / the reranker only calls
        this when ``loss.exec`` / ``score_mode=="exec"`` is active, so legacy
        paths never reach the raise (spec §2.2 C-10).
        """
        if self.exec_head is None:
            raise RuntimeError(
                "TextJEPA.predict_output called but exec_head is None "
                "(train with loss.exec.enabled=true / a wm_exec config)"
            )
        return self.exec_head.predict_output(pred_emb)

    def match_outputs(self, o_hat, z_out):
        """Output-equality logit between ô and an output embedding (spec §2.2).

        ``z_out`` must already be in proj_dim (``exec_head.embed_output``).
        """
        if self.exec_head is None:
            raise RuntimeError(
                "TextJEPA.match_outputs called but exec_head is None"
            )
        return self.exec_head.match_logit(o_hat, z_out)

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
