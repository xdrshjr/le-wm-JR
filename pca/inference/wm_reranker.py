"""WMReranker — learned world-model reranker (true predict-before-act).

Loads a ``TextJEPA`` (+ optional ``OutcomeHead``) trained at Stage-0 and
scores LLM candidate completions by **predicting** the post-execution
latent state — *without running any candidate code*. This is the
predict-before-act core of PCA (master-plan §G3): the model rolls out
the consequence of ``run_test`` in latent space and ranks candidates by
their predicted outcome.

Two scoring modes (spec §2.3):
    verifier : sigmoid(OutcomeHead(ẑ₁))                  — recommended main
    goal_dist: −‖ẑ₁ − z_goal‖², z_goal = encode(SUCCESS_TEMPLATE)

``serialize`` / ``SUCCESS_TEMPLATE`` / ``format_result`` are the single
source of truth shared by the trajectory collector
(``scripts/gen_mbpp_traj.py``) and this reranker, so training and
inference observe the same text distribution (spec §5.1, §6.4).

Cross-directory import note (spec §F5): callers under the repo-root
``scripts/`` must ``sys.path.insert(0, "<repo>/le-wm-JR")`` before
``from pca.inference.wm_reranker import WMReranker, serialize,
SUCCESS_TEMPLATE``. This module also self-inserts ``le-wm-JR`` for
direct execution.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# le-wm-JR root on sys.path so ``pca.*`` resolves even on direct import.
_LEWM_ROOT = Path(__file__).resolve().parents[2]
if str(_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEWM_ROOT))

from pca.action.schema import RunTestArgs  # noqa: E402

# Canonical "all visible tests passed" result string (goal_dist anchor).
SUCCESS_TEMPLATE = (
    "RESULT: passed all visible asserts; status=PASS; first_error: none"
)

# Selector for the single run_test op every reranker / trajectory uses.
_VISIBLE_SELECTOR = "visible_tests"
_TIMEOUT_SEC = 5


def serialize(prompt: str, program: str) -> str:
    """Build the pre-execution observation text for one candidate.

    Shared verbatim by the MBPP collector and the reranker so the world
    model sees the same ``obs_text`` distribution at train and test time.
    """
    return (
        f"PROBLEM:\n{prompt}\n"
        f"CANDIDATE:\n{program}\n"
        f"ACTION: run visible tests"
    )


def format_result(
    n_pass: int, n_total: int, first_error: str | None = None
) -> str:
    """Build the post-execution result text (step1 ``obs_text``).

    Returns ``SUCCESS_TEMPLATE`` verbatim on a full pass so the goal_dist
    anchor matches passing trajectories exactly (spec §5.1).
    """
    if n_total > 0 and n_pass >= n_total:
        return SUCCESS_TEMPLATE
    err = (first_error or "none").replace("\n", " ").strip()[:200] or "none"
    return (
        f"RESULT: passed {n_pass}/{n_total} visible asserts; "
        f"status=FAIL; first_error: {err}"
    )


def _run_test_op() -> RunTestArgs:
    return RunTestArgs(selector=_VISIBLE_SELECTOR, timeout_sec=_TIMEOUT_SEC)


@dataclass
class WMRerankerConfig:
    """Reranker configuration (spec §3.5)."""

    wm_config_name: str = "wm_humaneval"  # hydra config under config/train/
    ckpt_path: str | None = None
    score_mode: str = "verifier"  # "verifier" | "goal_dist"
    device: str = "cuda"
    max_obs_chars: int = 4000


def _maybe_autocast(device: torch.device):
    """fp16 mixed-precision on CUDA (Turing R01); no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class WMReranker:
    """Score candidate programs by latent-space outcome prediction.

    The selection stage performs **no** subprocess / ``exec`` / sandbox
    work — only a forward pass — which is the implementation-level proof
    of "predict-before-act" (spec §6.3a).
    """

    def __init__(self, cfg: WMRerankerConfig) -> None:
        self.cfg = cfg
        if cfg.score_mode not in ("verifier", "goal_dist"):
            raise ValueError(
                f"score_mode must be verifier|goal_dist, got {cfg.score_mode!r}"
            )
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model()
        self._load_ckpt()
        self.model.to(self.device).eval()
        self._z_goal: torch.Tensor | None = None
        if cfg.score_mode == "goal_dist":
            self._z_goal = self._encode([SUCCESS_TEMPLATE])  # (1, D)

    # -- construction ---------------------------------------------------

    def _build_model(self):
        from hydra.utils import instantiate

        cfg = self._compose_cfg()
        return instantiate(cfg.model)

    def _compose_cfg(self):
        """Compose the hydra config for ``wm_config_name``.

        Retries with ``return_hydra_config=True`` so a launcher default
        that overrides the ``hydra/launcher`` group (``# @package
        _global_``) composes cleanly outside ``@hydra.main``. Only
        ``cfg.model`` is consumed downstream.
        """
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        cfg_dir = str((_LEWM_ROOT / "config" / "train").resolve())
        for return_hydra in (False, True):
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            try:
                with initialize_config_dir(
                    version_base=None, config_dir=cfg_dir
                ):
                    return compose(
                        config_name=self.cfg.wm_config_name,
                        return_hydra_config=return_hydra,
                    )
            except Exception:
                if return_hydra:
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def _load_ckpt(self) -> None:
        cfg = self.cfg
        if not cfg.ckpt_path:
            if cfg.score_mode == "verifier":
                raise RuntimeError(
                    "verifier mode requires --wm-ckpt with a trained "
                    "outcome_head; none was provided"
                )
            print("[wm_reranker] no ckpt — random-init goal_dist")
            return
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(
            f"[wm_reranker] loaded {cfg.ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if cfg.score_mode == "verifier":
            self._assert_head_present(state, missing)

    def _assert_head_present(self, state: dict, missing) -> None:
        """F8: refuse to silently degrade verifier → goal_dist."""
        head_in_ckpt = any(
            k.startswith("outcome_head") for k in state.keys()
        )
        head_missing = any(
            str(k).startswith("outcome_head") for k in missing
        )
        if getattr(self.model, "outcome_head", None) is None:
            raise RuntimeError(
                "verifier mode but model has no outcome_head submodule — "
                "use a wm_config with outcome_head or pass "
                "--wm-score goal_dist"
            )
        if not head_in_ckpt or head_missing:
            raise RuntimeError(
                "verifier mode but checkpoint lacks trained outcome_head.* "
                "weights; refusing to silently fall back to goal_dist. "
                "Train with loss.outcome.enabled=true or pass "
                "--wm-score goal_dist explicitly."
            )

    # -- encoding / prediction -----------------------------------------

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode + predict ẑ₁ for raw observation texts. Returns (N, D)."""
        cap = self.cfg.max_obs_chars
        info = {
            "obs_text": [[t[:cap]] for t in texts],
            "op": [[_run_test_op()] for _ in texts],
        }
        with torch.no_grad(), _maybe_autocast(self.device):
            info = self.model.encode(info)
            emb = info["emb"]
            act_emb = info["act_emb"]
            z1 = self.model.predict(emb[:, :1], act_emb[:, :1])[:, -1]
        return z1.float()

    def score_candidates(
        self, prompt: str, programs: list[str]
    ) -> list[float]:
        """Predicted pass score per candidate (higher = more likely pass).

        Executes nothing: a single batched forward over the K candidates.
        """
        if not programs:
            return []
        texts = [serialize(prompt, prog) for prog in programs]
        z1 = self._encode(texts)  # (K, D)
        if self.cfg.score_mode == "verifier":
            logits = self.model.outcome_head(z1).squeeze(-1)  # (K,)
            scores = torch.sigmoid(logits)
        else:  # goal_dist
            diff = z1 - self._z_goal  # (K, D) broadcast over (1, D)
            scores = -(diff.pow(2).sum(dim=-1))
        return scores.detach().cpu().tolist()
