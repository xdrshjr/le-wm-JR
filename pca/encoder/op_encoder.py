"""OpEncoder — op_type one-hot + Qwen-tokenized arg MLP → R^out_dim.

Spec §2.2 + §4.1 + R05/R06: ``out_dim`` defaults to ``wm.embed_dim=384``;
constructor takes a single ``cfg`` dataclass to stay under
``MAX_FUNCTION_PARAMS=5``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from pydantic import BaseModel
from torch import nn

from pca.action.schema import OP_TYPES, ExecutableOp


@dataclass
class OpEncoderConfig:
    num_op_types: int = len(OP_TYPES)
    arg_max_tokens: int = 256
    out_dim: int = 384
    hidden_dim: int = 384
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-1.5B"


_OP_TYPE_TO_IDX = {name: i for i, name in enumerate(OP_TYPES)}


class OpEncoder(nn.Module):
    """Embed an ``ExecutableOp`` into a single vector in R^out_dim."""

    def __init__(self, cfg: OpEncoderConfig) -> None:
        super().__init__()
        from pca.utils.tokenizer import get_shared_tokenizer

        self.cfg = cfg
        self.tokenizer = get_shared_tokenizer(cfg.tokenizer_name)
        vocab = max(self.tokenizer.vocab_size, len(self.tokenizer))

        self.type_embed = nn.Embedding(cfg.num_op_types, cfg.hidden_dim)
        self.arg_token_embed = nn.Embedding(
            vocab, cfg.hidden_dim, padding_idx=self.tokenizer.pad_token_id
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.out_dim),
        )

    @staticmethod
    def _serialize_args(op: ExecutableOp) -> str:
        payload = op.model_dump() if isinstance(op, BaseModel) else dict(op)
        payload.pop("op_type", None)
        return json.dumps(payload, sort_keys=True, default=str)

    def _encode_one(
        self, ops: list[ExecutableOp]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        type_ids: list[int] = []
        arg_strs: list[str] = []
        for op in ops:
            op_type = op.op_type if isinstance(op, BaseModel) else op["op_type"]
            type_ids.append(_OP_TYPE_TO_IDX[op_type])
            arg_strs.append(self._serialize_args(op))

        type_tensor = torch.tensor(type_ids, dtype=torch.long, device=device)
        enc = self.tokenizer(
            arg_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.arg_max_tokens,
        )
        return type_tensor, enc["input_ids"].to(device)

    def forward(self, ops: list[ExecutableOp]) -> torch.Tensor:
        if len(ops) == 0:
            return torch.zeros(0, self.cfg.out_dim, device=self._device)

        type_ids, arg_ids = self._encode_one(ops)
        type_vec = self.type_embed(type_ids)  # (B, hidden)
        arg_vec = self.arg_token_embed(arg_ids).mean(dim=1)  # (B, hidden)
        return self.mlp(torch.cat([type_vec, arg_vec], dim=-1))

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device
