"""ExecutableOp schema — 8 op types covering SWE-agent operations.

Phase B'.1 (T03.1) from
``docs/plans/world-model-llm-coding-fusion/specs/03-action-space.md``,
consumed by ``pca.encoder.op_encoder.OpEncoder``.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

OP_TYPES = (
    "apply_patch",
    "run_test",
    "search_code",
    "edit_file",
    "read_file",
    "ls",
    "cd",
    "git_diff",
)

OpType = Literal[
    "apply_patch",
    "run_test",
    "search_code",
    "edit_file",
    "read_file",
    "ls",
    "cd",
    "git_diff",
]


class _OpArgsBase(BaseModel):
    model_config = {"extra": "forbid"}


class ApplyPatchArgs(_OpArgsBase):
    op_type: Literal["apply_patch"] = "apply_patch"
    diff: str
    target_path: str | None = None


class RunTestArgs(_OpArgsBase):
    op_type: Literal["run_test"] = "run_test"
    selector: str = ""
    timeout_sec: int = 300


class SearchCodeArgs(_OpArgsBase):
    op_type: Literal["search_code"] = "search_code"
    query: str
    path: str = "."
    regex: bool = False


class EditFileArgs(_OpArgsBase):
    op_type: Literal["edit_file"] = "edit_file"
    path: str
    start_line: int = 1
    end_line: int | None = None
    new_text: str = ""


class ReadFileArgs(_OpArgsBase):
    op_type: Literal["read_file"] = "read_file"
    path: str
    start_line: int = 1
    end_line: int | None = None


class LsArgs(_OpArgsBase):
    op_type: Literal["ls"] = "ls"
    path: str = "."
    depth: int = 1


class CdArgs(_OpArgsBase):
    op_type: Literal["cd"] = "cd"
    path: str


class GitDiffArgs(_OpArgsBase):
    op_type: Literal["git_diff"] = "git_diff"
    ref: str = "HEAD"
    path: str | None = None


ExecutableOp = Annotated[
    Union[
        ApplyPatchArgs,
        RunTestArgs,
        SearchCodeArgs,
        EditFileArgs,
        ReadFileArgs,
        LsArgs,
        CdArgs,
        GitDiffArgs,
    ],
    Field(discriminator="op_type"),
]
