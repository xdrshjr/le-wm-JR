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


def _repair_user(problem: str, body: str, failing: list | None = None) -> str:
    cases = ""
    if failing:
        joined = "\n".join(failing[:5])
        cases = (
            "\nThe world model predicts it FAILS these specific tests — fix "
            f"them in particular:\n{joined}\n"
        )
    return (
        "This function body is predicted to fail its tests. Return a "
        "corrected function body:\n\n"
        f"```python\n{problem}{body}\n```\n{cases}"
    )


def _parse_io(tests: list) -> tuple[list, list]:
    """Asserts → ``(call_forms, expecteds)``, dropping unparseable ones."""
    from pca.inference.consensus import parse_assert_io

    calls, expecteds = [], []
    for t in tests:
        call, expected = parse_assert_io(t)
        if call is None:
            continue
        calls.append(call)
        expecteds.append(expected)
    return calls, expecteds


def _majority_rep(col: list[str]) -> "str | None":
    """Raw decoded string of the column's largest NON-ERROR normalised
    cluster; ``None`` when only crash clusters exist — a ``<<ERR>>``
    representative would steer repair toward the very cluster the R2
    invariant zero-scores (review fix R9)."""
    from collections import Counter

    from pca.inference.consensus import norm_repr

    norms = [norm_repr(d) for d in col]
    counts = Counter(n for n in norms if not n.startswith("<<ERR>>"))
    if not counts:
        return None
    top = counts.most_common(1)[0][0]
    return col[norms.index(top)]


def _interp_hints(matrix: list, decoded: list, io: tuple, best: int,
                  thr: float) -> list[str]:
    """Top-1's predicted-failing cells → human-readable repair triples
    (spec §2.5 D-1 step 2: "on input x your code returns ŷ; expected e").

    ``io = (calls, expecteds)``; ``expecteds is None`` = no_doctest — the
    hint compares against the majority cluster's representative instead.
    """
    from pca.inference.consensus import norm_repr

    calls, exps = io
    hints: list[str] = []
    for t, call in enumerate(calls):
        y_hat = decoded[best][t]
        if exps is not None and exps[t] is not None:
            if (matrix[best][t] < thr
                    and norm_repr(y_hat) != norm_repr(exps[t])):
                # the EM guard kills the degenerate "returns X; expected X"
                # hint a low em_weight can produce (review fix R9)
                hints.append(f"On input {call} your code returns {y_hat}; "
                             f"expected {exps[t]}")
        else:
            col_best = max(matrix[c][t] for c in range(len(matrix)))
            if matrix[best][t] < col_best:
                rep = _majority_rep([decoded[c][t]
                                     for c in range(len(decoded))])
                if rep is None or norm_repr(rep) == norm_repr(y_hat):
                    continue  # only-ERR / tied clusters — no useful hint
                hints.append(f"On input {call} your code returns {y_hat}; "
                             f"most candidates return {rep}")
    return hints[:5]


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
        # PEC closed-loop knobs (spec §3 P1 / §2.4). Set as attributes (not
        # ctor params) to keep the signature ≤5; callers may override before
        # calling ``repair_and_rerank_pec``.
        self.consensus_mode = "soft_conf"
        self.gamma = 1.0
        self.theta = 0.5

    def _verifier_prob(self, prompt: str, programs: list[str]) -> torch.Tensor:
        z1 = self.aligned.predict_outcome_latent(prompt, programs)
        return self.aligned._verifier_scores(z1)

    def predicted_failing_tests(
        self, prompt: str, program: str, tests: list[str], *, thr: float = 0.5,
    ) -> list[str]:
        """Tests the world model PREDICTS ``program`` fails (zero exec; spec
        §3 P1 stretch). Reuses the PEC matrix
        (``AlignedWMLLM.predict_pass_matrix`` — execution-derived when the
        aligned model is in exec_mode, spec §2.4.3) so repair can be steered by
        the concrete predicted-failing cases rather than a single global scalar.
        """
        if not tests:
            return []
        mat = self.aligned.predict_pass_matrix(prompt, [program], tests)
        if mat.numel() == 0:
            return []
        probs = mat[0].tolist()
        return [t for t, p in zip(tests, probs) if p < thr]

    def _failing_io_hints(self, failing: list[str]) -> list[str]:
        """Frame predicted-failing asserts as (input → expected) repair hints.

        R8 (spec §2.4.3): the execution world model predicts the candidate's
        output on each input; for the inputs where that predicted output ≠ the
        expected value, surface ``input → expected`` so the LLM repairs the
        exact computation the WM imagines wrong (predict-act-predict).
        """
        from pca.inference.consensus import parse_assert_io

        hints: list[str] = []
        for t in failing:
            call, expected = parse_assert_io(t)
            if call is None:
                hints.append(t)
            elif expected is not None:
                hints.append(
                    f"{call} should return {expected} "
                    "(world model predicts your output differs)"
                )
            else:
                hints.append(f"{call} (world model predicts a wrong output)")
        return hints

    def _generate_repairs(
        self, problem: str, body: str, failing: list | None = None,
    ) -> list[str]:
        tok = self.aligned.tokenizer
        chat = tok.apply_chat_template(
            [{"role": "system", "content": _REPAIR_SYS},
             {"role": "user", "content": _repair_user(problem, body, failing)}],
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

    # ----- round-9 interp closed loop (spec §2.5 D-1; zero execution) ---

    def _gen_repairs_via(self, interp, problem: str, body: str,
                         hints: list[str]) -> list[str]:
        """Repair generation through the interpreter's shared LLM — inside
        ``generation_ctx()`` so the executor-LoRA is DISABLED (inv. 6)."""
        tok = interp.tok
        chat = tok.apply_chat_template(
            [{"role": "system", "content": _REPAIR_SYS},
             {"role": "user", "content": _repair_user(problem, body, hints)}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(chat, return_tensors="pt").to(interp.device)
        do_sample = self.temperature > 0
        with torch.no_grad(), interp.generation_ctx():
            out = interp.model.generate(
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

    def repair_and_rerank_interp(self, prompt: str, cands: list, interp,
                                 tests: list, cfg: dict) -> dict:
        """Interp closed loop (spec §2.5 D-1): rank by the DISCRETE interp
        consensus; take the top-1's predicted-failing cells; repair with
        the first HUMAN-READABLE imagined execution ("on input x your code
        returns ŷ; expected e"); re-rank the merged pool. Zero execution.

        ``cfg`` keys: ``calib`` (InterpCalib), ``theta``/``mode``/``gamma``
        (consensus), ``theta_fail`` (cell threshold), ``has_doctest``
        (TRUE stratum from the bench — proposer-guessed expecteds are
        never trusted; review fix R9), ``pick`` (optional tie-break-chain
        picker), ``assemble`` (prompt, text) → Candidate. Returns a dict
        whose ``total_budget`` (= K + R) feeds the mandatory equal-budget
        SC@(K+R) comparison (spec §2.5 D-1 step 4 — without it the gain
        may NOT be reported).
        """
        from pca.inference.consensus import InterpCalib, consensus_rank

        calls, expecteds = _parse_io(tests or [])
        logprobs = [c.logprob for c in cands]
        base = {"repaired": False, "total_budget": len(cands)}
        if not calls:
            return {**base, "pick": cands[self._argmax(logprobs)],
                    "scores": []}
        exps = expecteds if cfg.get("has_doctest") else None
        calib = cfg.get("calib") or InterpCalib()
        kw = {"theta": cfg.get("theta", 0.5),
              "mode": cfg.get("mode", "soft_conf"),
              "gamma": cfg.get("gamma", 1.0)}

        def _pick(pool, scores):
            picker = cfg.get("pick")
            if picker is not None:
                return picker(pool, scores)[0]
            return pool[self._argmax(scores)]

        mat, dec = interp.matrix(prompt, [c.program for c in cands], calls,
                                 exps, calib=calib)
        scores = consensus_rank(mat, logprobs, **kw)
        best = self._argmax(scores)
        thr = cfg.get("theta_fail", self.repair_threshold)
        hints = _interp_hints(mat, dec, (calls, exps), best, thr)
        if not hints:
            return {**base, "pick": _pick(cands, scores), "scores": scores}
        texts = self._gen_repairs_via(interp, prompt, cands[best].text,
                                      hints)
        merged = list(cands) + [cfg["assemble"](prompt, t) for t in texts]
        mat2, _ = interp.matrix(prompt, [c.program for c in merged], calls,
                                exps, calib=calib)
        scores2 = consensus_rank(mat2, [c.logprob for c in merged], **kw)
        return {"pick": _pick(merged, scores2), "scores": scores2,
                "repaired": True, "n_hints": len(hints),
                "total_budget": len(cands) + len(texts)}

    def _pec_scores(
        self, prompt: str, programs: list[str], tests: list[str],
    ) -> list[float]:
        """Zero-exec CodeT consensus over the predicted (K,T) matrix."""
        from pca.inference.consensus import consensus_rank

        mat = self.aligned.predict_pass_matrix(prompt, programs, tests)
        return consensus_rank(
            mat.tolist(), [0.0] * len(programs), theta=self.theta,
            mode=self.consensus_mode, gamma=self.gamma,
        )

    def repair_and_rerank_pec(
        self, prompt: str, cands: list, assemble, tests: list[str],
    ) -> tuple:
        """PEC closed loop (spec §3 P1 / §6 fusion evidence; zero execution).

        Rank by CodeT consensus over the predicted ``(K, T)`` matrix; if the
        top candidate is predicted to fail some asserts, repair it **steered by
        those concrete predicted-failing tests** (``predicted_failing_tests``),
        resample, and re-rank the enlarged set by PEC. This closes the
        predict→act→predict loop and routes the WM's per-test prediction into
        the LLM's repair decision — the direct LLM↔WM coupling the round-3 red
        line asks for. Falls back to log-prob argmax when there are no visible
        tests. Nothing is executed.
        """
        programs = [c.program for c in cands]
        if not tests:
            return cands[self._argmax([c.logprob for c in cands])], []
        scores = self._pec_scores(prompt, programs, tests)
        best = self._argmax(scores)
        failing = self.predicted_failing_tests(
            prompt, cands[best].program, tests, thr=self.repair_threshold,
        )
        if not failing:
            return cands[best], scores
        hints = self._failing_io_hints(failing)
        repaired = self._generate_repairs(prompt, cands[best].text, hints)
        merged = list(cands) + [assemble(prompt, r) for r in repaired]
        new_scores = self._pec_scores(
            prompt, [c.program for c in merged], tests,
        )
        return merged[self._argmax(new_scores)], new_scores
