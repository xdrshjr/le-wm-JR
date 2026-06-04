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


# R3 (spec §2.4): the discriminative ``TEST:`` segment is placed BEFORE the
# (often long) ``CANDIDATE:`` segment, and the candidate is capped, so that
# under the MiniLM ~512-token / ``max_obs_chars`` window only the candidate
# tail (lowest-information) can be truncated — PROBLEM/TEST/ACTION always
# survive. ``serialize_test`` is the single source of truth shared verbatim
# by ``gen_mbpp_traj`` (training) and PEC inference (spec §2.4 invariant 2).
_PROG_CAP = 1600  # chars; PROBLEM/TEST are short, leave headroom for candidate


def serialize_test(prompt: str, program: str, test: str) -> str:
    """Per-test observation: predict whether ``program`` passes ``test``.

    Field order PROBLEM→TEST→CANDIDATE→ACTION keeps the discriminative TEST
    segment truncation-safe (spec §2.4 / R3). Shared verbatim by the MBPP
    per-test collector and the PEC reranker (train == test distribution).
    """
    prog = program if len(program) <= _PROG_CAP else program[:_PROG_CAP]
    return (
        f"PROBLEM:\n{prompt}\n"
        f"TEST:\n{test}\n"
        f"CANDIDATE:\n{prog}\n"
        f"ACTION: run this test"
    )


# R8 (spec wm-exec-trace-fusion-sota §2.2): execution-trace observation. The
# world model predicts the candidate's OUTPUT on a test INPUT (not a pass bit),
# so the field order is PROBLEM→INPUT→CANDIDATE→ACTION — the discriminative
# INPUT segment stays before the (capped) candidate, truncation-safe like
# ``serialize_test`` (R3). ``test_input`` is the call form only (e.g.
# ``f(2,3)``) — NO expected value (that is the other input to the comparison;
# spec §2.2). ``serialize_exec`` / ``serialize_output`` are the single source
# of truth shared verbatim by ``gen_exec_traj`` (training) and
# ``score_matrix_exec`` (inference).


def serialize_exec(problem: str, candidate: str, test_input: str) -> str:
    """Per-input observation: predict ``candidate`` output on ``test_input``."""
    prog = candidate if len(candidate) <= _PROG_CAP else candidate[:_PROG_CAP]
    return (
        f"PROBLEM:\n{problem}\n"
        f"INPUT:\n{test_input}\n"
        f"CANDIDATE:\n{prog}\n"
        f"ACTION: predict output"
    )


def serialize_output(output_repr: str) -> str:
    """Post-execution output observation (``repr``-normalised value text)."""
    return f"OUTPUT:\n{output_repr}"


# Canonical single-test "passed" result string (per-test step1 obs_text).
PASS_TEMPLATE = "RESULT: test passed; status=PASS; first_error: none"


def format_test_result(passed: bool, err: str | None = None) -> str:
    """Build the post-execution result text for a single test (spec §2.4)."""
    if passed:
        return PASS_TEMPLATE
    e = (err or "none").replace("\n", " ").strip()[:200] or "none"
    return f"RESULT: test failed; status=FAIL; first_error: {e}"


def _run_test_op() -> RunTestArgs:
    return RunTestArgs(selector=_VISIBLE_SELECTOR, timeout_sec=_TIMEOUT_SEC)


@dataclass
class WMRerankerConfig:
    """Reranker configuration (spec §3.5)."""

    wm_config_name: str = "wm_humaneval"  # hydra config under config/train/
    ckpt_path: str | None = None
    score_mode: str = "verifier"  # "verifier" | "goal_dist" | "exec"
    device: str = "cuda"
    max_obs_chars: int = 4000
    verifier_temp: float = 1.0  # sigmoid temperature for score_matrix (PEC)
    # R8 exec mode (spec §2.4.1): output-similarity temperature + cluster
    # threshold for the execution-derived consensus matrix.
    exec_tau: float = 1.0
    exec_cluster_thr: float = 0.7


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
        if cfg.score_mode not in ("verifier", "goal_dist", "exec"):
            raise ValueError(
                "score_mode must be verifier|goal_dist|exec, got "
                f"{cfg.score_mode!r}"
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
        elif cfg.score_mode == "exec":
            self._assert_exec_head_present(state, missing)

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

    def _assert_exec_head_present(self, state: dict, missing) -> None:
        """R8 (spec §2.5 C-1): refuse to silently drop exec_head weights.

        Mirrors ``_assert_head_present`` — if the config has no exec_head or
        the checkpoint lacks trained ``exec_head.*`` weights (e.g. the old
        ``wm_humaneval`` config was passed by mistake), raise instead of
        evaluating an untrained head.
        """
        head_in_ckpt = any(k.startswith("exec_head") for k in state.keys())
        head_missing = any(str(k).startswith("exec_head") for k in missing)
        if getattr(self.model, "exec_head", None) is None:
            raise RuntimeError(
                "exec mode but model has no exec_head submodule — use the "
                "wm_exec_humaneval config (NOT wm_humaneval), which silently "
                "drops exec_head weights under strict=False."
            )
        if not head_in_ckpt or head_missing:
            raise RuntimeError(
                "exec mode but checkpoint lacks trained exec_head.* weights; "
                "refusing to evaluate an untrained head. Train Stage-0E with "
                "loss.exec.enabled=true (config wm_exec_humaneval)."
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

    def score_matrix(
        self, prompt: str, programs: list[str], tests: list[str],
        *, temp: float | None = None, return_logits: bool = False,
    ) -> "torch.Tensor":
        """(K, T) predicted P(candidate passes test); **zero execution**.

        For every ``(program, test)`` pair build the per-test observation
        via ``serialize_test`` (TEST-before-CANDIDATE, truncation-safe; R3)
        and run a single batched ``_encode`` + OutcomeHead forward. Empty
        ``tests`` → ``(K, 0)`` so the caller falls back to logprob argmax
        (spec §2.2(b) / §4.2). ``return_logits`` yields raw pre-temperature
        logits (calibration); otherwise ``sigmoid(logit / temp)``.
        """
        k, t = len(programs), len(tests)
        if k == 0 or t == 0:
            return torch.zeros((k, t))
        head = getattr(self.model, "outcome_head", None)
        if head is None:
            raise RuntimeError("score_matrix requires a trained outcome_head")
        texts = [
            serialize_test(prompt, prog, tst)
            for prog in programs for tst in tests
        ]
        z1 = self._encode(texts)  # (K*T, D)
        logits = head(z1).squeeze(-1).view(k, t)  # (K, T)
        if return_logits:
            return logits.detach().cpu()
        tv = self.cfg.verifier_temp if temp is None else temp
        tv = max(float(tv), 1e-3)
        return torch.sigmoid(logits / tv).detach().cpu()

    def _encode_obs(self, texts: list[str]) -> torch.Tensor:
        """Encoder embedding of raw observation texts (NO predict). (N, D).

        For output texts (``serialize_output``) we want the encoder's view of
        the value, matching the training target ``tgt_emb`` (the encoded
        ``obs_next``), not a predicted next-state latent (spec §2.2 ô↔ẑ₁).
        """
        cap = self.cfg.max_obs_chars
        info = {
            "obs_text": [[t[:cap]] for t in texts],
            "op": [[_run_test_op()] for _ in texts],
        }
        with torch.no_grad(), _maybe_autocast(self.device):
            info = self.model.encode(info)
            emb = info["emb"][:, 0]
        return emb.float()

    def exec_embeddings(
        self, prompt: str, programs: list[str], inputs: list[str],
        expected: list[str] | None = None,
    ):
        """Predicted output embeddings for the exec PEC matrix; zero execution.

        Returns ``(o_hat (K,T,P) cpu, z_exp (T,P) cpu | None)``. Split out so
        calibration can cache the embeddings once and sweep τ / cluster_thr in
        pure Python (spec §2.4.1). ``expected`` given → also encode the expected
        outputs; else ``z_exp`` is ``None`` (consistency path).
        """
        k, t = len(programs), len(inputs)
        head = getattr(self.model, "exec_head", None)
        if head is None:
            raise RuntimeError("exec_embeddings requires a trained exec_head")
        texts = [
            serialize_exec(prompt, prog, inp)
            for prog in programs for inp in inputs
        ]
        with torch.no_grad():
            o_hat = head.predict_output(self._encode(texts)).view(k, t, -1)
            z_exp = None
            if expected is not None:
                emb = self._encode_obs(
                    [serialize_output(e or "") for e in expected]
                )
                z_exp = head.embed_output(emb).detach().cpu()
        return o_hat.detach().cpu(), z_exp

    def score_matrix_exec(
        self, prompt: str, programs: list[str], tests: list[str],
        *, expected: list[str] | None = None, tau: float | None = None,
    ) -> "torch.Tensor":
        """(K, T) execution-derived pass/consistency matrix; **zero execution**.

        Predicts each candidate's output embedding ô(c,t) via ``serialize_exec``
        + ``exec_head.predict_output``; the matrix is derived by the single
        ``consensus.exec_pass_from_outputs`` (spec §2.4.1 C-8): ``expected``
        given (has_doctest) → compare predicted output to the expected output;
        else → candidate-vs-candidate output consistency. Empty ``tests`` →
        ``(K, 0)`` so the caller falls back to log-prob argmax.
        """
        from pca.inference.consensus import exec_pass_from_outputs

        if len(programs) == 0 or len(tests) == 0:
            return torch.zeros((len(programs), len(tests)))
        o_hat, z_exp = self.exec_embeddings(prompt, programs, tests, expected)
        tau_v = self.cfg.exec_tau if tau is None else tau
        matrix = exec_pass_from_outputs(
            o_hat, z_exp, tau=tau_v, cluster_thr=self.cfg.exec_cluster_thr,
        )
        return torch.tensor(matrix)
