"""AlignedWMLLM — LLaVA-style aligned world model + projector + LLM.

Inference-side object shared by the two WM→LLM reasoning paths
(wm-llm-alignment spec §2.2 / §4.2):

  path A  ``generate_conditioned`` — prepend the projected world-model
          outcome prediction (soft tokens) to the LLM context and sample
          candidates *conditioned on the predicted consequence* (the most
          faithful LLaVA mapping; improves "what to generate").
  path B  ``score_candidate`` — fused reranking score = verifier ×
          aligned-conditional-likelihood (auto-gated by ``n_doctests``),
          no candidate is executed (improves "which to pick").

Composition: a **frozen** ``TextJEPA`` (the LLaVA "vision encoder"), a
``WorldModelProjector`` (K soft tokens, the LLaVA projection ``W``), and
``Qwen2.5-1.5B-Instruct`` optionally wrapped with a LoRA adapter.

Honesty invariant (spec §6): the selection / generation stages execute
**no** candidate code — only forward passes. Real execution happens only
offline at training-label time. HumanEval is graded by the caller on the
hidden test.

``aligned_ckpt`` is the Stage-2 product directory (spec §3 R6):
``projector.pt`` + ``lora/`` (PEFT ``save_pretrained``) + ``outcome_head.pt``
(+ optional ``align_config.json``). A plain ``.pt`` file is also accepted
(treated as a combined state dict) for forward compatibility.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

# le-wm-JR root on sys.path so ``pca.*`` resolves on direct import (spec §F5).
_LEWM_ROOT = Path(__file__).resolve().parents[1]
if str(_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEWM_ROOT))

from pca.action.schema import RunTestArgs  # noqa: E402
from pca.inference.wm_reranker import (  # noqa: E402
    SUCCESS_TEMPLATE,
    serialize,
)
from pca.projector.mlp import (  # noqa: E402
    WorldModelProjector,
    WorldModelProjectorConfig,
)

_VISIBLE_SELECTOR = "visible_tests"
_TIMEOUT_SEC = 5
_DEFAULT_K = 4  # soft tokens when no aligned_ckpt pins K (spec §2.3).

# Same chat framing as scripts/bench_humaneval._format_chat so conditioned
# candidates stay on the base method's distribution (only the soft prefix
# differs). Kept here verbatim to keep this module self-contained.
_SYS_MSG = (
    "You are an expert Python programmer. "
    "Complete the given function. "
    "Output ONLY the function body, no explanation, no markdown, "
    "no extra definitions. Indent every line with 4 spaces."
)


def _user_msg(problem_prompt: str) -> str:
    return (
        "Complete this Python function. "
        "Return only the function body (no `def` line, no imports, "
        "no markdown fence):\n\n"
        f"```python\n{problem_prompt}```\n"
    )


def _run_test_op() -> RunTestArgs:
    return RunTestArgs(selector=_VISIBLE_SELECTOR, timeout_sec=_TIMEOUT_SEC)


def _autocast(device: torch.device):
    """fp16 mixed precision on CUDA (Turing R01); no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


class AlignedWMLLM(nn.Module):
    """Aligned WM + projector + LLM; see module docstring for the contract."""

    def __init__(
        self,
        wm_cfg_name: str,
        wm_ckpt: str,
        llm_name: str,
        aligned_ckpt: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        # Fusion / conditioning knobs (caller overrides post-construction;
        # tuned only on an independent val set — spec §2.2 / §6 red line).
        self.alpha = 0.5            # verifier weight when doctests present
        self.cond_signal = "both"   # "z1" | "goal" | "both" (ablation e)
        self.w_self_test = 0.0      # P1 self-proposed-test term (off)
        self.test_proposer = None
        self.max_obs_chars = 4000

        self.d_wm = 384
        self.wm = self._build_wm(wm_cfg_name)
        if wm_ckpt:
            self._load_wm_ckpt(wm_ckpt)
        _freeze(self.wm)
        self.wm.to(self.device).eval()

        self.llm, self.tokenizer = self._build_llm(llm_name)
        self.d_llm = self.llm.get_input_embeddings().weight.size(1)

        self.num_tokens = self._infer_num_tokens(aligned_ckpt)
        self.projector = self._build_projector()
        if aligned_ckpt:
            self._load_aligned(aligned_ckpt)
        _freeze(self.projector)
        self.projector.to(self.device).eval()

        self._z_goal = self._encode_latent([SUCCESS_TEMPLATE]).detach()

    # -- construction ---------------------------------------------------

    def _build_wm(self, wm_cfg_name: str):
        from hydra.utils import instantiate

        cfg = self._compose_cfg(wm_cfg_name)
        self.d_wm = int(cfg.wm.embed_dim)
        return instantiate(cfg.model)

    def _compose_cfg(self, wm_cfg_name: str):
        """Compose ``config/train/<wm_cfg_name>`` outside ``@hydra.main``.

        Mirrors ``WMReranker._compose_cfg``: retry with
        ``return_hydra_config=True`` so a ``# @package _global_`` launcher
        default composes cleanly. Only ``cfg.model`` / ``cfg.wm`` are used.
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
                        config_name=wm_cfg_name,
                        return_hydra_config=return_hydra,
                    )
            except Exception:
                if return_hydra:
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def _load_wm_ckpt(self, wm_ckpt: str) -> None:
        state = torch.load(wm_ckpt, map_location="cpu")
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        self.wm.load_state_dict(state, strict=False)

    def _build_llm(self, llm_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=dtype, trust_remote_code=True
        )
        _freeze(llm)
        return llm.to(self.device).eval(), tok

    def _build_projector(self) -> WorldModelProjector:
        dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        cfg = WorldModelProjectorConfig(
            in_dim=self.d_wm,
            out_dim=self.d_llm,
            num_tokens=self.num_tokens,
            dtype=dtype,
        )
        return WorldModelProjector(cfg)

    def _infer_num_tokens(self, aligned_ckpt: str | None) -> int:
        """K = projector.pt ``fc2`` rows / out_dim, else the default."""
        if not aligned_ckpt:
            return _DEFAULT_K
        sd = self._read_projector_state(aligned_ckpt)
        w = sd.get("fc2.weight") if sd else None
        if w is None:
            return _DEFAULT_K
        return max(1, int(w.shape[0]) // self.d_llm)

    @staticmethod
    def _read_projector_state(aligned_ckpt: str) -> dict | None:
        p = Path(aligned_ckpt)
        if p.is_dir():
            p = p / "projector.pt"
        if not p.exists():
            return None
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict):
            return state.get("projector", state)
        return None

    def _load_aligned(self, aligned_ckpt: str) -> None:
        """Load Stage-2 product: projector + LoRA + outcome_head (spec R6)."""
        d = Path(aligned_ckpt)
        proj = self._read_projector_state(aligned_ckpt)
        if proj is not None:
            self.projector.load_state_dict(proj, strict=False)
        cfg_path = d / "align_config.json" if d.is_dir() else None
        if cfg_path is not None and cfg_path.exists():
            meta = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.cond_signal = meta.get("cond_signal", self.cond_signal)
        self._load_lora(d)
        self._load_outcome_head(d)

    def _load_lora(self, d: Path) -> None:
        lora = d / "lora" if d.is_dir() else None
        if lora is None or not lora.exists():
            return
        from peft import PeftModel

        self.llm = PeftModel.from_pretrained(self.llm, str(lora))
        _freeze(self.llm)
        self.llm.to(self.device).eval()

    def _load_outcome_head(self, d: Path) -> None:
        head_pt = d / "outcome_head.pt" if d.is_dir() else None
        head = getattr(self.wm, "outcome_head", None)
        if head_pt is None or not head_pt.exists() or head is None:
            return
        head.load_state_dict(torch.load(head_pt, map_location="cpu"))

    # -- world-model encoding -------------------------------------------

    def _encode_latent(self, texts: list[str]) -> torch.Tensor:
        """Encode + predict ẑ₁ for raw observation texts → (N, d_wm)."""
        cap = self.max_obs_chars
        info = {
            "obs_text": [[t[:cap]] for t in texts],
            "op": [[_run_test_op()] for _ in texts],
        }
        with torch.no_grad(), _autocast(self.device):
            info = self.wm.encode(info)
            emb, act = info["emb"], info["act_emb"]
            z1 = self.wm.predict(emb[:, :1], act[:, :1])[:, -1]
        return z1.float()

    def predict_outcome_latent(
        self, prompt: str, programs: list[str]
    ) -> torch.Tensor:
        """Predicted post-execution latent per candidate → (N, d_wm)."""
        texts = [serialize(prompt, p) for p in programs]
        return self._encode_latent(texts)

    def soft_tokens(self, z_pred: torch.Tensor) -> torch.Tensor:
        """Project latents to K LLM-space soft tokens → (N, K, d_llm)."""
        return self.projector(z_pred.to(self.device))

    def _cond_soft(self, z1: torch.Tensor) -> torch.Tensor:
        """Soft tokens for one item, mixing ẑ₁ and the goal latent.

        ``cond_signal`` selects the conditioning (ablation e, spec §6):
        ``z1`` (draft consequence only), ``goal`` (success anchor only),
        ``both`` (⌈K/2⌉ from ẑ₁ + the rest from the goal latent).
        """
        if self.cond_signal == "z1":
            return self.projector(z1.to(self.device))
        goal = self._z_goal.to(self.device)
        if self.cond_signal == "goal":
            return self.projector(goal)
        s1 = self.projector(z1.to(self.device))
        s2 = self.projector(goal)
        half = (self.num_tokens + 1) // 2
        return torch.cat([s1[:, :half], s2[:, : self.num_tokens - half]], dim=1)

    # -- path A: predict-conditioned generation -------------------------

    def _format_chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYS_MSG},
             {"role": "user", "content": _user_msg(prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prefix_embeds(self, prompt: str, soft: torch.Tensor):
        """[chat(problem) embeds ⊕ soft] → (inputs_embeds, attn, L, K)."""
        embed = self.llm.get_input_embeddings()
        chat = self._format_chat(prompt)
        ids = self.tokenizer(chat, return_tensors="pt").input_ids
        ids = ids.to(self.device)
        p_emb = embed(ids)
        soft = soft.to(p_emb.dtype)
        inp = torch.cat([p_emb, soft], dim=1)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
        return inp, attn, ids.size(1), soft.size(1)

    def generate_conditioned(
        self,
        prompt: str,
        draft: str,
        *,
        k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[tuple[str, float]]:
        """Path A — sample ``k`` candidates conditioned on the WM soft
        tokens. Executes nothing. Returns ``[(text, mean_logprob)]``.
        """
        z1 = self.predict_outcome_latent(prompt, [draft])  # (1, d_wm)
        soft = self._cond_soft(z1)                          # (1, K, d_llm)
        inp, attn, _l, _k = self._prefix_embeds(prompt, soft)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.llm.generate(
                inputs_embeds=inp,
                attention_mask=attn,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        return self._decode_logprob(out, k)

    def _decode_logprob(self, out, k: int) -> list[tuple[str, float]]:
        seqs = out.sequences  # (k, L_new); inputs_embeds → generated only
        scores = out.scores   # tuple len L_new of (k, vocab)
        n_new = len(scores)
        if seqs.size(1) > n_new:  # robustness across transformers versions
            seqs = seqs[:, -n_new:]
        eos = self.tokenizer.eos_token_id
        results: list[tuple[str, float]] = []
        for i in range(k):
            toks = seqs[i]
            logps: list[float] = []
            for t, s in enumerate(scores):
                if t >= toks.size(0):
                    break
                tid = toks[t].item()
                if tid == eos:
                    break
                lp = torch.log_softmax(s[i].float(), dim=-1)
                logps.append(lp[tid].item())
            mean_lp = sum(logps) / max(1, len(logps))
            text = self.tokenizer.decode(toks, skip_special_tokens=True)
            results.append((text, mean_lp))
        return results

    # -- path B: fused predictive rerank --------------------------------

    def _verifier_scores(self, z1: torch.Tensor) -> torch.Tensor:
        head = getattr(self.wm, "outcome_head", None)
        if head is None:
            return torch.zeros(z1.size(0))
        logit = head(z1).squeeze(-1)
        return torch.sigmoid(logit).float().cpu()

    def _cond_logprob(
        self, prompt: str, completion: str, soft: torch.Tensor
    ) -> float:
        """Mean per-token logp of ``completion`` under [chat ⊕ soft]."""
        embed = self.llm.get_input_embeddings()
        chat = self._format_chat(prompt)
        p_ids = self.tokenizer(chat, return_tensors="pt").input_ids
        c_ids = self.tokenizer(
            completion, return_tensors="pt", add_special_tokens=False
        ).input_ids
        p_ids, c_ids = p_ids.to(self.device), c_ids.to(self.device)
        if c_ids.size(1) == 0:
            return 0.0
        p_emb, c_emb = embed(p_ids), embed(c_ids)
        soft = soft.to(p_emb.dtype)
        inp = torch.cat([p_emb, soft, c_emb], dim=1)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.llm(inputs_embeds=inp, attention_mask=attn).logits[0]
        start = p_ids.size(1) + soft.size(1)
        m = c_ids.size(1)
        sel = logits[start - 1: start - 1 + m].float()
        lp = torch.log_softmax(sel, dim=-1)
        tok_lp = lp[torch.arange(m, device=lp.device), c_ids[0]]
        return float(tok_lp.mean().item())

    def _likelihood_scores(
        self, prompt: str, completions: list[str], z1: torch.Tensor
    ) -> torch.Tensor:
        lps = [
            self._cond_logprob(prompt, c, self._cond_soft(z1[i: i + 1]))
            for i, c in enumerate(completions)
        ]
        return _zscore(torch.tensor(lps, dtype=torch.float32))

    def _self_test_scores(
        self, prompt: str, programs: list[str]
    ) -> torch.Tensor:
        """P1 (spec §2.2 ``w_t``): predict pass on LLM self-proposed tests."""
        if self.w_self_test <= 0 or self.test_proposer is None:
            return torch.zeros(len(programs))
        tests = self.test_proposer.propose(prompt)
        if not tests:
            return torch.zeros(len(programs))
        suffix = "\n" + "\n".join(tests)
        z1 = self.predict_outcome_latent(prompt, [p + suffix for p in programs])
        return self._verifier_scores(z1)

    def score_candidate(
        self,
        prompt: str,
        programs: list[str],
        n_doctests: int = 0,
        completions: list[str] | None = None,
    ) -> list[float]:
        """Path B — fused score per candidate (no execution).

        ``score = w_v·verifier + w_l·zscore(cond_logp) + w_t·self_test``,
        auto-gated by ``n_doctests`` (spec §2.2): with visible doctests the
        verifier term gets weight ``alpha``; without, the score falls back
        entirely to the dense conditional likelihood. ``completions`` are
        the raw candidate texts whose likelihood is scored (defaults to the
        assembled ``programs`` when not supplied).
        """
        if not programs:
            return []
        z1 = self.predict_outcome_latent(prompt, programs)  # (N, d_wm)
        verifier = self._verifier_scores(z1)
        texts = completions if completions is not None else programs
        likelihood = self._likelihood_scores(prompt, texts, z1)
        self_test = self._self_test_scores(prompt, programs)
        if n_doctests > 0:
            w_v, w_l = self.alpha, 1.0 - self.alpha
        else:
            w_v, w_l = 0.0, 1.0
        score = w_v * verifier + w_l * likelihood + self.w_self_test * self_test
        return score.tolist()
