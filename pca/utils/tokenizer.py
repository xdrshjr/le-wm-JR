"""Shared Qwen tokenizer accessor (cached singleton).

The boundary is intentional (RA3): CodeBERT is used *only* inside
``ToolIOEncoder``; everything else — including ``OpEncoder`` arg
tokenization — uses the Qwen tokenizer so that downstream LLM
prefixing has consistent token IDs.
"""
from __future__ import annotations

from functools import lru_cache

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-1.5B"


@lru_cache(maxsize=4)
def get_shared_tokenizer(name: str = DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
