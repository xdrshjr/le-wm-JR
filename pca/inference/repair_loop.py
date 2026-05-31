"""RepairLoop — predict→repair→re-rank (zero execution; spec §3 P1 stretch).

The fused reranker (`AlignedWMLLM.score_candidate`) scores every candidate by
a learned world model's **predicted** P(pass) without running anything. When
the top predicted candidate is itself likely to fail, this loop asks the LLM
to *repair* the worst candidate — guided only by the WM's prediction, never by
a real run — then re-scores the enlarged set with the same predict-before-act
fused score and returns the new argmax.

Honesty invariant (spec §6): nothing is executed at any point — repair is a
forward generation conditioned on the predicted failure, and re-ranking is the
same fused forward score. Decoupled from ``bench_humaneval`` via an injected
``assemble(prompt, text) -> program`` callable, so this module imports no
benchmark code.
"""
from __future__ import annotations

import torch

_REPAIR_SYS = (
    "You are an expert Python programmer fixing a buggy function body. "
    "A world model predicts the current body will FAIL its tests. Output "
    "ONLY the corrected function body — no def line, no imports, no markdown "
    "— indented with 4 spaces."
)


def _repair_user(problem: str, body: str) -> str:
    return (
        "This function body is predicted to fail its tests. Return a "
        "corrected function body:\n\n"
        f"```python\n{problem}{body}\n```\n"
    )


class RepairLoop:
    """Predict→repair→re-rank with the aligned WM (executes nothing)."""

    def __init__(
        self, aligned, *,
        repair_threshold: float = 0.5, n_repair: int = 2,
        max_new_tokens: int = 512, temperature: float = 0.7,
    ) -> None:
        self.aligned = aligned
        self.repair_threshold = repair_threshold
        self.n_repair = n_repair
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _verifier_prob(self, prompt: str, programs: list[str]) -> torch.Tensor:
        z1 = self.aligned.predict_outcome_latent(prompt, programs)
        return self.aligned._verifier_scores(z1)

    def _generate_repairs(self, problem: str, body: str) -> list[str]:
        tok = self.aligned.tokenizer
        chat = tok.apply_chat_template(
            [{"role": "system", "content": _REPAIR_SYS},
             {"role": "user", "content": _repair_user(problem, body)}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(chat, return_tensors="pt").to(self.aligned.llm.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            out = self.aligned.llm.generate(
                **inputs, do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                num_return_sequences=self.n_repair,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        return [
            tok.decode(o[in_len:], skip_special_tokens=True) for o in out
        ]

    @staticmethod
    def _argmax(scores: list[float]) -> int:
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def repair_and_rerank(
        self, prompt: str, cands: list, assemble, n_doctests: int = 0,
    ) -> tuple:
        """Return ``(best_cand, scores)``; repair the worst candidate when
        the top one is predicted to fail. ``cands`` are duck-typed bench
        ``Candidate``s (``.program`` / ``.text``); ``assemble(prompt, text)``
        rebuilds an executable-shaped program from a repaired body. Nothing
        is executed (spec §6).
        """
        scores = self.aligned.score_candidate(
            prompt, [c.program for c in cands], n_doctests,
            [c.text for c in cands],
        )
        best = self._argmax(scores)
        prob = float(self._verifier_prob(prompt, [cands[best].program])[0])
        if prob >= self.repair_threshold:
            return cands[best], scores
        worst = int(min(range(len(scores)), key=lambda i: scores[i]))
        repaired = self._generate_repairs(prompt, cands[worst].text)
        merged = list(cands) + [assemble(prompt, r) for r in repaired]
        new_scores = self.aligned.score_candidate(
            prompt, [c.program for c in merged], n_doctests,
            [c.text for c in merged],
        )
        return merged[self._argmax(new_scores)], new_scores
