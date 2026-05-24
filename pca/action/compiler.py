"""NLIntentCompiler — heuristic NL → ExecutableOp.

Implements Phase B'.2 (T03.4) from
``docs/plans/world-model-llm-coding-fusion/todos/03-action-space.md``.
The LLM-fallback path is a hook for adaptation-spec v2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from pca.action.schema import (
    ApplyPatchArgs,
    CdArgs,
    EditFileArgs,
    ExecutableOp,
    GitDiffArgs,
    LsArgs,
    ReadFileArgs,
    RunTestArgs,
    SearchCodeArgs,
)

_OP_ADAPTER: TypeAdapter[ExecutableOp] = TypeAdapter(ExecutableOp)


@dataclass(frozen=True)
class NeedsClarification:
    why: str


_RUN_TEST = re.compile(r"^(run|pytest|test)\b\s*(?P<sel>.*)$", re.IGNORECASE)
_SEARCH = re.compile(
    r"^(grep|search|find)\b\s+(?P<q>.+?)(?:\s+in\s+(?P<path>\S+))?$",
    re.IGNORECASE,
)
_READ = re.compile(r"^(cat|read|open)\b\s+(?P<path>\S+)$", re.IGNORECASE)
_LS = re.compile(r"^(ls|list)\b\s*(?P<path>\S+)?$", re.IGNORECASE)
_CD = re.compile(r"^cd\b\s+(?P<path>\S+)$", re.IGNORECASE)
_GIT_DIFF = re.compile(
    r"^(git\s+diff|diff)\b\s*(?P<ref>\S+)?$", re.IGNORECASE
)
_EDIT = re.compile(
    r"^edit\b\s+(?P<path>\S+)(?:\s+lines?\s+(?P<lo>\d+)"
    r"(?:[-:](?P<hi>\d+))?)?$",
    re.IGNORECASE,
)


class NLIntentCompiler:
    """Compile a free-form NL intent string into an ExecutableOp.

    Returns ``NeedsClarification`` when the pattern is ambiguous; callers
    are expected to re-prompt the LLM (≤3 retries — see TODO 03.6).
    """

    def compile(
        self, intent_text: str
    ) -> ExecutableOp | NeedsClarification:
        text = (intent_text or "").strip()
        if not text:
            return NeedsClarification("empty intent")

        if text.lstrip().startswith(("---", "diff --git", "+++ ")):
            return ApplyPatchArgs(diff=text)

        m = _RUN_TEST.match(text)
        if m:
            return RunTestArgs(selector=m.group("sel").strip())

        m = _SEARCH.match(text)
        if m:
            return SearchCodeArgs(
                query=m.group("q").strip(),
                path=m.group("path") or ".",
            )

        m = _READ.match(text)
        if m:
            return ReadFileArgs(path=m.group("path"))

        m = _EDIT.match(text)
        if m:
            lo = int(m.group("lo")) if m.group("lo") else 1
            hi_raw = m.group("hi")
            hi = int(hi_raw) if hi_raw else None
            return EditFileArgs(
                path=m.group("path"), start_line=lo, end_line=hi
            )

        m = _LS.match(text)
        if m:
            return LsArgs(path=m.group("path") or ".")

        m = _CD.match(text)
        if m:
            return CdArgs(path=m.group("path"))

        m = _GIT_DIFF.match(text)
        if m:
            return GitDiffArgs(ref=m.group("ref") or "HEAD")

        return NeedsClarification(f"no pattern matched: {text[:60]!r}")

    def validate(self, op_dict: dict) -> ExecutableOp | NeedsClarification:
        try:
            return _OP_ADAPTER.validate_python(op_dict)
        except ValidationError as exc:
            return NeedsClarification(str(exc))
