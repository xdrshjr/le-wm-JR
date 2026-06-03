"""ToolIOEncoder — frozen LM base + transformer head → R^out_dim.

Spec §2.2 + §4.1 + R06/R07. Single ``cfg`` parameter keeps ctor at
2 args incl. self (under MAX_FUNCTION_PARAMS=5).

Two encoder shapes, selected by ``cfg`` (backward compatible — the MVP /
goal_dist configs keep the original CLS + length-1-head path bit-for-bit):

  legacy (``pooling='cls'``, ``head_over_tokens=False``):
      base(frozen) → [CLS] hidden → Linear → head(seq-len-1) → Linear
  token-head (``pooling='mean'``, ``head_over_tokens=True``):
      base(frozen) → per-token hidden → Linear → head over the *full*
      token sequence (padding-masked) → masked-mean pool → Linear

The token-head path is what the HumanEval verifier uses: it lets the
trainable head attend over code tokens (the discriminative signal) and
mean-pools the way sentence-/code-LM bases are actually trained, instead
of reading a single near-meaningless CLS vector (Round-3 fix).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ToolIOEncoderConfig:
    base_model: str = "microsoft/codebert-base"
    hidden_dim: int = 384
    out_dim: int = 384
    num_head_layers: int = 4
    freeze_base: bool = True
    max_length: int = 512
    # Round-3 (backward-compatible) knobs. Defaults reproduce the legacy
    # CLS + seq-len-1-head behaviour, so MVP / goal_dist configs are
    # unchanged; the HumanEval verifier config opts into mean + token-head.
    pooling: str = "cls"            # "cls" | "mean"
    head_over_tokens: bool = False  # run head over full token sequence
    # Round-7 (spec §2.2 lever A) — "true" code-native world model: let the
    # top of the frozen base actually learn execution semantics instead of
    # only borrowing a generic sentence vector. All default to OFF, so every
    # pre-R7 config (MVP / goal_dist / R6 verifier) is byte-for-byte unchanged.
    unfreeze_top_n: int = 0      # unfreeze the base's top-n transformer layers
    use_lora: bool = False       # inject LoRA on the base (mutually exclusive)
    lora_rank: int = 8
    lora_alpha: int = 16
    encoder_dropout: float = 0.0  # dropout after the head (anti-overfit)


class ToolIOEncoder(nn.Module):
    """Encode a batch of free-form tool-IO text strings into R^out_dim."""

    def __init__(self, cfg: ToolIOEncoderConfig) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModel.from_pretrained(cfg.base_model)
        base_hidden = self.base.config.hidden_size

        if cfg.unfreeze_top_n > 0 and cfg.use_lora:
            raise ValueError(
                "unfreeze_top_n and use_lora are mutually exclusive "
                "(spec §2.2): set exactly one of them."
            )
        if cfg.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.base.eval()
        # Round-7 lever A: controlled unfreeze / LoRA on top of the freeze.
        if cfg.unfreeze_top_n > 0:
            self._apply_unfreeze(cfg.unfreeze_top_n)
        elif cfg.use_lora:
            self._apply_lora(cfg)

        self.input_proj = nn.Linear(base_hidden, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=8,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.head = nn.TransformerEncoder(layer, num_layers=cfg.num_head_layers)
        self.dropout = nn.Dropout(cfg.encoder_dropout)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        # Any trainable base param ⇒ the base forward must build a graph.
        self._base_trainable = any(
            p.requires_grad for p in self.base.parameters()
        )

    def _base_layers(self):
        """Locate the base transformer's layer list + final norm (R7 P1-2).

        ``AutoModel.from_pretrained`` on Qwen2.5 returns a ``Qwen2Model``
        whose decoder layers live at ``base.layers`` and final norm at
        ``base.norm`` — NOT ``base.model.layers`` (that is the
        ``Qwen2ForCausalLM`` path, absent here). Probe order: ``base.layers``
        (Qwen2, first) → ``base.model.layers`` (version fallback) →
        ``base.encoder.layer`` (codebert). Returns ``(layers, final_norm)``
        with ``final_norm=None`` when the branch has no obvious末层 LN.
        """
        base = self.base
        if hasattr(base, "layers"):
            return base.layers, getattr(base, "norm", None)
        inner = getattr(base, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers, getattr(inner, "norm", None)
        enc = getattr(base, "encoder", None)
        if enc is not None and hasattr(enc, "layer"):
            return enc.layer, None
        return None, None

    def _apply_unfreeze(self, n: int) -> None:
        """Unfreeze the top-n base layers (+ final LN) so they can learn."""
        layers, final_norm = self._base_layers()
        if layers is None:
            raise RuntimeError(
                "unfreeze_top_n>0 but no transformer layer list found "
                "(tried base.layers / base.model.layers / base.encoder.layer)"
            )
        for layer in list(layers)[-min(n, len(layers)):]:
            for p in layer.parameters():
                p.requires_grad_(True)
            layer.train()
        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad_(True)
            final_norm.train()

    def _apply_lora(self, cfg) -> None:
        """Inject LoRA adapters on the base (low-overfit alt to unfreeze)."""
        from peft import LoraConfig, get_peft_model

        from pca.training.llava_stages import _LORA_TARGETS

        lora_cfg = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
            target_modules=list(_LORA_TARGETS), lora_dropout=0.0,
            bias="none",
        )
        self.base = get_peft_model(self.base, lora_cfg)
        self.base.enable_input_require_grads()

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        return {k: v.to(device) for k, v in enc.items()}

    def _base_hidden(self, enc: dict) -> torch.Tensor:
        # When the base is fully frozen, ``_base_trainable`` is False and
        # ``set_grad_enabled(False)`` is exactly the old ``no_grad`` path
        # (zero regression). With R7 top-n unfreeze / LoRA, gradients must
        # flow through the base, so the graph is built (spec §2.2 P1-1).
        with torch.set_grad_enabled(self._base_trainable):
            out = self.base(**enc)
        return out.last_hidden_state  # (B, L, base_hidden)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over valid tokens. x:(B,L,D) mask:(B,L) -> (B,D)."""
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros(0, self.cfg.out_dim, device=self._device)

        enc = self._tokenize(texts)
        hidden = self._base_hidden(enc)  # (B, L, base_hidden)
        mask = enc["attention_mask"]      # (B, L)

        if self.cfg.head_over_tokens:
            x = self.input_proj(hidden)              # (B, L, hidden)
            pad = mask == 0                          # (B, L) True = pad
            x = self.head(x, src_key_padding_mask=pad)
            pooled = self._masked_mean(x, mask)      # (B, hidden)
        else:
            if self.cfg.pooling == "mean":
                base_vec = self._masked_mean(hidden, mask)  # (B, base_hidden)
            else:  # legacy CLS
                base_vec = hidden[:, 0]
            x = self.input_proj(base_vec).unsqueeze(1)  # (B, 1, hidden)
            x = self.head(x)
            pooled = x[:, 0]
        return self.output_proj(self.dropout(pooled))  # (B, out_dim)

    def state_dict(self, *args, **kwargs):
        """Drop the *frozen* base LM weights from checkpoints (R3 + R7).

        The base is reloaded from HF at construction, so persisting its
        frozen weights every epoch is pure waste — a Qwen-0.5B base adds
        ~2 GB per checkpoint and filled the disk. R7 refinement (spec §2.2):
        only ``requires_grad=False`` base params (and all base buffers, which
        HF reconstructs) are dropped — the unfrozen top-n layers / LoRA
        adapters are KEPT so the learned execution semantics survive the
        checkpoint. ``load_state_dict(..., strict=False)`` then leaves the
        still-frozen base at its pretrained init (correct) and restores the
        trainable layers/adapters. Trainable layers add only tens of MB
        (<400 MB total ckpt; spec §5 / dev-log records the size).
        """
        sd = super().state_dict(*args, **kwargs)
        if not self.cfg.freeze_base:
            return sd
        prefix = kwargs.get("prefix", "")
        if not prefix and len(args) >= 2 and isinstance(args[1], str):
            prefix = args[1]
        drop = self._frozen_base_keys(prefix)
        for k in [k for k in sd if k in drop]:
            del sd[k]
        return sd

    def _frozen_base_keys(self, prefix: str) -> set:
        """state_dict keys to drop: frozen base params + all base buffers."""
        root = f"{prefix}base."
        keep = {
            f"{root}{name}"
            for name, p in self.base.named_parameters()
            if p.requires_grad
        }
        all_base = {
            f"{root}{name}" for name, _ in self.base.named_parameters()
        }
        all_base |= {
            f"{root}{name}" for name, _ in self.base.named_buffers()
        }
        return all_base - keep

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device
