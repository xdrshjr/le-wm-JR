"""TestProposer — LLM self-proposed tests (CodeT-style; wm-llm-alignment P1).

Asks the generator LLM to propose a few ``assert`` statements for a coding
problem, so the fused reranker can add a verifier term on the *predicted*
pass of those self-proposed tests (``w_t`` in spec §2.2) — giving the
``no_doctest`` subset a verifier signal it otherwise lacks.

Honesty invariant: the proposed asserts are **predicted** by the world
model (``OutcomeHead(ẑ₁(program; proposed_tests))``), never executed here.
The proposer only generates text; ``AlignedWMLLM`` consumes it.

Off by default (``AlignedWMLLM.w_self_test = 0``); attach an instance and
set ``w_self_test > 0`` to enable.
"""
from __future__ import annotations

import re

_ASSERT_LINE = re.compile(r"^\s*assert\b.*")

_SYS_MSG = (
    "You are a meticulous Python test engineer. Given a function "
    "signature and docstring, output a few standalone `assert` statements "
    "that any correct implementation must pass. Output ONLY assert lines, "
    "one per line, no prose, no markdown fence."
)


def _user_msg(prompt: str) -> str:
    return (
        "Write up to {n} assert statements (calling the function by its "
        "name) for this problem:\n\n```python\n{p}```\n"
    ).format(n="{n}", p=prompt)


class TestProposer:
    """Generate ``assert`` tests for a problem via the generator LLM."""

    def __init__(
        self, tokenizer, model, *,
        n_tests: int = 3, max_new_tokens: int = 128, temperature: float = 0.6,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.n_tests = n_tests
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _chat(self, prompt: str) -> str:
        user = _user_msg(prompt).replace("{n}", str(self.n_tests))
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYS_MSG},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def propose(self, prompt: str) -> list[str]:
        """Return up to ``n_tests`` proposed assert statements (no exec)."""
        import torch

        chat = self._chat(prompt)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return self._parse_asserts(text)

    def _parse_asserts(self, text: str) -> list[str]:
        asserts: list[str] = []
        for line in text.splitlines():
            if _ASSERT_LINE.match(line):
                asserts.append(line.strip())
            if len(asserts) >= self.n_tests:
                break
        return asserts
