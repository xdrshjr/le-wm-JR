"""PCA training-stage modules.

LLaVA-style two-stage WM↔LLM alignment lives here so ``train_pca.py``
stays a thin dispatcher (spec §3, wm-llm-alignment). ``llava_stages``
holds the Stage-1 feature-alignment and Stage-2 instruction-tuning
Lightning modules plus their loss helpers.
"""
from pca.training.llava_stages import (
    AlignStage1Module,
    InstructStage2Module,
)

__all__ = ["AlignStage1Module", "InstructStage2Module"]
