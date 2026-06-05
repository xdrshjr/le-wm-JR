"""NeuralInterpreter — generator-hosted executor-LoRA (round 9, spec §2.2).

The world model lives INSIDE the generator LLM as a peft LoRA adapter
("weight-level fusion", spec §2.1): the same Qwen2.5-1.5B-Instruct writes
code with the adapter DISABLED and imagines execution with it ENABLED.
Given ``serialize_exec(problem, candidate, test_input)`` it autoregressively
generates ``serialize_output(output_repr)``; P(pass) derives from the
expected-output conditional log-likelihood + greedy exact match, candidate
consistency from normalised decoded-string equality (``consensus.py``).

Zero-execution invariant (spec §2.1 inv. 1): every method here is a pure
forward pass (teacher-forced log-prob or greedy decode) — no candidate is
ever run.

Invariant 6 (LoRA leak guard, spec R-12): the adapter's DEFAULT state is
DISABLED — it is enabled only inside the scoring methods and restored in a
``finally``, so any generation through the shared model (the bench holds
the same underlying module) is byte-identical to the un-adapted base.

Token contract (spec §2.1 inv. 2, review C-1): targets get ``eos_token``
appended at TOKENISATION time (never written to jsonl); decoding stops at
EOS and comparisons take the ``OUTPUT:`` block body; over-long prompts are
shrunk by ``shrink_prompt`` (shared verbatim with ``train_executor``) —
implicit tokenizer left-truncation is never relied on.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

# le-wm-JR root on sys.path so ``pca.*`` resolves even on direct import.
_LEWM_ROOT = Path(__file__).resolve().parents[2]
if str(_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEWM_ROOT))

from pca.inference.consensus import (  # noqa: E402
    InterpCalib,
    interp_pass_from_strings,
    norm_repr,
)
from pca.inference.wm_reranker import (  # noqa: E402
    serialize_exec,
    serialize_output,
)

_CAND_MARK = "\nCANDIDATE:\n"
_ACT_MARK = "\nACTION:"
_OUT_MARK = "OUTPUT:"
_MIN_CAND_CHARS = 64
# Token-position budget per scoring batch (dynamic batching, spec §5).
_BATCH_TOKEN_BUDGET = 16384


def encode_prompt(tok, text: str) -> list[int]:
    """Prompt token ids — shared by SFT and inference (inv. 2, C-1)."""
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_target(tok, text: str) -> list[int]:
    """Target token ids + EOS terminator (appended at tokenise time only;
    never part of the ``serialize_output`` text — review C-1 ①)."""
    return tok(text, add_special_tokens=False)["input_ids"] + \
        [tok.eos_token_id]


def shrink_prompt(tok, text: str, budget: int) -> tuple[str, bool]:
    """Fit ``text`` into ``budget`` tokens by shrinking ONLY the CANDIDATE
    segment (inv. 2 ③) — the same rule at train and inference time, so
    the model never sees an implicitly left-truncated observation.

    Returns ``(text, truncated)``; callers count ``truncated`` into the
    train manifest (>5% → raise ``max_len``; spec §2.2).
    """
    if len(encode_prompt(tok, text)) <= budget:
        return text, False
    if _CAND_MARK not in text or _ACT_MARK not in text:
        ids = encode_prompt(tok, text)[:budget]
        return tok.decode(ids), True
    head, rest = text.split(_CAND_MARK, 1)
    cand, tail = rest.rsplit(_ACT_MARK, 1)
    out = text
    while (len(encode_prompt(tok, out)) > budget
           and len(cand) > _MIN_CAND_CHARS):
        cand = cand[: max(_MIN_CAND_CHARS, int(len(cand) * 0.8))]
        out = head + _CAND_MARK + cand + _ACT_MARK + tail
    if len(encode_prompt(tok, out)) > budget:  # head/tail alone too long
        out = tok.decode(encode_prompt(tok, out)[:budget])
    return out, True


def extract_output_block(text: str) -> str:
    """Decoded text → ``OUTPUT:`` block body (inv. 2 ②): everything after
    the first ``OUTPUT:`` marker (EOS already stripped by the decode)."""
    idx = text.find(_OUT_MARK)
    body = text[idx + len(_OUT_MARK):] if idx >= 0 else text
    return body.strip()


def _manifest_max_len(adapter_dir: "str | None") -> "int | None":
    """``max_len`` from the adapter's train_manifest.json (the manifest
    lives beside the snapshot dirs) so inference shares the exact training
    prompt budget (inv. 2 ③ — the >5% remedy raises max_len to 1280)."""
    if not adapter_dir:
        return None
    d = Path(adapter_dir).resolve()
    for spot in (d / "train_manifest.json", d.parent / "train_manifest.json"):
        try:
            return int(json.loads(spot.read_text(encoding="utf-8"))
                       ["max_len"])
        except (OSError, ValueError, KeyError):
            continue
    return None


class NeuralInterpreter:
    """Generator LLM + executor-LoRA zero-execution neural interpreter.

    Wraps the bench's ALREADY-LOADED tokenizer/model (weight sharing —
    it never builds its own copy). ``adapter_dir=None`` runs the prompted
    base model (the G0 zero-shot probe, spec §6).
    """

    def __init__(self, tok, model, adapter_dir: str | None,
                 max_new: int = 96) -> None:
        self.tok = tok
        if tok.pad_token_id is None:  # local guard (entry points also set it)
            tok.pad_token = tok.eos_token
        self.max_new = max_new
        # Keep the training-time prompt budget (inv. 2): read max_len from
        # the adapter's train_manifest.json when available, else 1024.
        self.max_len = _manifest_max_len(adapter_dir) or 1024
        self.adapter_dir = adapter_dir
        self._peft = adapter_dir is not None
        if self._peft:
            self.model = self._wrap_adapter(model, adapter_dir)
        else:
            self.model = model
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _wrap_adapter(model, adapter_dir: str):
        """In-place ``PeftModel`` wrap; adapter starts DISABLED (inv. 6)."""
        try:
            from peft import PeftModel  # noqa: PLC0415 — peft ≥0.10 (R-14)
        except ImportError as e:  # pragma: no cover - env precheck handles
            raise RuntimeError(
                "peft>=0.10 is required for --executor-lora; install it "
                "first (spec §2.6 step 0 / R-14)"
            ) from e
        wrapped = PeftModel.from_pretrained(
            model, adapter_dir, is_trainable=False
        )
        base_dtype = next(model.parameters()).dtype
        for name, p in wrapped.named_parameters():
            if "lora" in name and p.dtype != base_dtype:
                p.data = p.data.to(base_dtype)
        wrapped.eval()
        wrapped.disable_adapter_layers()  # default = byte-identical base
        return wrapped

    @contextmanager
    def _adapter_on(self):
        """Enable the adapter for one scoring call; ALWAYS restore the
        disabled default (inv. 6 — generation must never see the LoRA)."""
        if not self._peft:
            yield
            return
        self.model.enable_adapter_layers()
        try:
            yield
        finally:
            self.model.disable_adapter_layers()

    @contextmanager
    def generation_ctx(self):
        """Adapter-disabled context for candidate generation (inv. 6).

        With the default-disabled policy this is a guard, not a switch —
        it re-asserts the disabled state so a crashed scoring call can
        never leak LoRA weights into base/SC sampling (R-12).
        """
        if self._peft:
            self.model.disable_adapter_layers()
        yield

    # ----- batching -----------------------------------------------------

    def _prompt_budget(self) -> int:
        return self.max_len - self.max_new

    def _fit(self, prompts: list[str]) -> list[str]:
        return [
            shrink_prompt(self.tok, p, self._prompt_budget())[0]
            for p in prompts
        ]

    @staticmethod
    def _batches(items: list, sizes: list[int]) -> list[list[int]]:
        """Index batches packed so ``n × max_len`` stays under budget."""
        out: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i, n in enumerate(sizes):
            new_max = max(cur_max, n)
            if cur and (len(cur) + 1) * new_max > _BATCH_TOKEN_BUDGET:
                out.append(cur)
                cur, cur_max = [], 0
                new_max = n
            cur.append(i)
            cur_max = new_max
        if cur:
            out.append(cur)
        return out

    # ----- scoring (pure forward; zero execution) ------------------------

    def _tf_batch(self, pairs: list[tuple[list[int], list[int]]]
                  ) -> list[float]:
        """Teacher-forced length-normalised log-prob of each target
        (EOS included in the average; review C-1)."""
        pad = self.tok.pad_token_id
        seqs = [p + t for p, t in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.tensor(
            [s + [pad] * (width - len(s)) for s in seqs], device=self.device
        )
        attn = torch.tensor(
            [[1] * len(s) + [0] * (width - len(s)) for s in seqs],
            device=self.device,
        )
        with torch.no_grad(), self._adapter_on():
            logits = self.model(input_ids=ids, attention_mask=attn).logits
        out: list[float] = []
        for i, (p, t) in enumerate(pairs):
            # Slice the target window BEFORE the fp32 log-softmax: the
            # full-batch (positions × vocab) fp32 copy would be ~10 GiB at
            # the 16k-token budget and OOM a 24 GB 3090 (review fix R9).
            pos = torch.log_softmax(
                logits[i, len(p) - 1: len(p) - 1 + len(t), :].float(),
                dim=-1,
            )
            tgt = torch.tensor(t, device=self.device).unsqueeze(-1)
            out.append(float(pos.gather(-1, tgt).mean().item()))
        return out

    def expected_logprobs(self, prompts: list[str],
                          expecteds: list[str]) -> list[float]:
        """lp̄ of ``serialize_output(e)`` given each prompt (one TF pass)."""
        return self._lp_fitted(self._fit(prompts), expecteds)

    def _lp_fitted(self, fitted: list[str],
                   expecteds: list[str]) -> list[float]:
        pairs = [
            (encode_prompt(self.tok, p),
             encode_target(self.tok, serialize_output(e)))
            for p, e in zip(fitted, expecteds)
        ]
        sizes = [len(p) + len(t) for p, t in pairs]
        out: list[float] = [0.0] * len(pairs)
        for batch in self._batches(pairs, sizes):
            vals = self._tf_batch([pairs[i] for i in batch])
            for i, v in zip(batch, vals):
                out[i] = v
        return out

    def _decode_batch(self, texts: list[str]) -> list[str]:
        old_side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            enc = self.tok(
                texts, return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad(), self._adapter_on():
                seq = self.model.generate(
                    **enc, do_sample=False, max_new_tokens=self.max_new,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
        finally:
            self.tok.padding_side = old_side
        new = seq[:, enc["input_ids"].shape[1]:]
        return [
            extract_output_block(
                self.tok.decode(row, skip_special_tokens=True)
            )
            for row in new
        ]

    def decode_outputs(self, prompts: list[str]) -> list[str]:
        """Greedy (temp=0, deterministic) OUTPUT block per prompt —
        the no_doctest consistency path + repair-hint ŷ source."""
        return self._decode_fitted(self._fit(prompts))

    def _decode_fitted(self, fitted: list[str]) -> list[str]:
        sizes = [len(encode_prompt(self.tok, p)) + self.max_new
                 for p in fitted]
        out: list[str] = [""] * len(fitted)
        for batch in self._batches(fitted, sizes):
            vals = self._decode_batch([fitted[i] for i in batch])
            for i, v in zip(batch, vals):
                out[i] = v
        return out

    def score_expected(self, prompts: list[str], expecteds: list[str]
                       ) -> list[tuple[float, bool]]:
        """Per sample ``(lp̄, em)``: one teacher-forced pass + one greedy
        decode (spec §2.2); ``em`` compares ``norm_repr`` forms. Prompts
        are fitted ONCE so both readings share the same observation."""
        fitted = self._fit(prompts)
        lps = self._lp_fitted(fitted, expecteds)
        decs = self._decode_fitted(fitted)
        return [
            (lp, norm_repr(d) == norm_repr(e))
            for lp, d, e in zip(lps, decs, expecteds)
        ]

    def matrix(self, problem: str, programs: list[str], calls: list[str],
               expecteds: list[str] | None = None, *,
               calib: InterpCalib = InterpCalib(),
               ) -> tuple[list[list[float]], list[list[str]]]:
        """(K,T) discrete pass matrix + decoded ŷ grid for one problem.

        Shared by the bench picker and the repair loop so both derive the
        matrix from the same prompts / decode / lp pipeline (inv. 2).
        """
        k, t = len(programs), len(calls)
        fitted = self._fit([
            serialize_exec(problem, prog, call)
            for prog in programs for call in calls
        ])
        decoded = self._decode_fitted(fitted)
        dec2 = [decoded[i * t:(i + 1) * t] for i in range(k)]
        lp2 = None
        if expecteds is not None:
            # Score ONLY the columns with an expected value: None columns
            # are clustered by ``interp_pass_from_strings`` and never read
            # their lp cell (mixed-column support; review fix R9).
            cols = [j for j, e in enumerate(expecteds) if e is not None]
            lp2 = [[0.0] * t for _ in range(k)]
            if cols:
                sub = [fitted[i * t + j] for i in range(k) for j in cols]
                exp = [expecteds[j] for _ in range(k) for j in cols]
                lps = self._lp_fitted(sub, exp)
                for n, (i, j) in enumerate(
                        (i, j) for i in range(k) for j in cols):
                    lp2[i][j] = lps[n]
        mat = interp_pass_from_strings(dec2, lp2, expecteds, calib=calib)
        return mat, dec2
