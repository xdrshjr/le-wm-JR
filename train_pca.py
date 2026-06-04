"""PCA training entry point (R03 + wm-llm-alignment §3).

Bypasses ``swm.data.load_dataset`` by consuming ``TrajectoryDataset``
directly. Dispatches on ``cfg.experiment.stage``:

    stage0  world model only (TextJEPA + OutcomeHead / R8 ExecTraceHead)
            — the R8 exec loss branch lives in ``pca.forward`` and is gated by
            ``loss.exec.enabled``; with the encoder fully frozen
            ``_stage0_optim`` keeps its single-group path (zero change here)
    stage1  LLaVA feature alignment (projector only)         — real loss
    stage2  predict-conditioned instruction tuning           — projector
            + Qwen-LoRA + OutcomeHead

Stage-1 replaces the old norm-matching stub with ``AlignStage1Module``
(InfoNCE + cosine); Stage-2 is new (``InstructStage2Module``). Freezing is
manual via ``requires_grad_(False)`` (R04 — Lightning 2.x has no ``Freeze``
callback). Stage products are written by ``StageCkptCallback`` (spec §3 R6):
``stage1/weights_epoch_{N}.pt`` (projector) and ``stage2/`` (projector.pt +
lora/ + outcome_head.pt + align_config.json).
"""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from module import SIGReg
from pca.data.collate import collate_trajectories
from pca.data.dataset import TrajectoryDataset, TrajectoryDatasetConfig
from pca.forward import pca_forward
from utils import SaveCkptCallback


def _build_dataset(cfg, split: str) -> TrajectoryDataset:
    ds_cfg = TrajectoryDatasetConfig(
        path=cfg.data.path,
        split=split,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        seed=cfg.seed,
        max_obs_chars=cfg.data.get("max_obs_chars", 16000),
    )
    return TrajectoryDataset(ds_cfg)


def _maybe_balanced_sampler(cfg, train_set):
    """Stage-0 pass/fail class-balancing sampler (spec §2.3.1); else None."""
    stage = cfg.experiment.get("stage", "stage0")
    if stage != "stage0" or not cfg.data.get("class_balance", False):
        return None
    from pca.training.llava_stages import class_balanced_sampler

    return class_balanced_sampler(train_set, seed=cfg.seed)


def _build_loaders(cfg) -> tuple:
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set = _build_dataset(cfg, "train")
    val_set = _build_dataset(cfg, "val")
    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    sampler = _maybe_balanced_sampler(cfg, train_set)
    train = torch.utils.data.DataLoader(
        train_set,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_trajectories,
        generator=rnd_gen,
        **loader_kwargs,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_trajectories,
        **loader_kwargs,
    )
    return train, val


def _freeze(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def _load_wm(cfg) -> torch.nn.Module:
    """Instantiate the frozen world model and load the Stage-0 ckpt."""
    world_model = hydra.utils.instantiate(cfg.model.wm)
    if cfg.model.get("wm_ckpt"):
        state = torch.load(cfg.model.wm_ckpt, map_location="cpu")
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        world_model.load_state_dict(state, strict=False)
    _freeze(world_model)
    return world_model


def _build_train_projector(cfg) -> torch.nn.Module:
    """Projector with fp32 master weights (AMP/16-mixed needs fp32)."""
    from pca.projector.mlp import (
        WorldModelProjector,
        WorldModelProjectorConfig,
    )

    proj = cfg.model.projector
    pcfg = WorldModelProjectorConfig(
        in_dim=cfg.wm.embed_dim,
        hidden_dim=proj.hidden_dim,
        out_dim=proj.out_dim,
        num_tokens=proj.get("num_tokens", 4),
        dtype=torch.float32,
    )
    return WorldModelProjector(pcfg)


def _has_trainable_base(world_model) -> bool:
    """True iff the encoder base has any trainable param (R7 unfreeze/LoRA)."""
    enc = getattr(world_model, "encoder", None)
    base = getattr(enc, "base", None) if enc is not None else None
    if base is None:
        return False
    return any(p.requires_grad for p in base.parameters())


def _stage0_optim(cfg, world_model) -> dict:
    """spt optimizer spec for Stage-0 (spec §2.2 P1-1, corrected落点 D1).

    Single group ``model_opt`` (legacy, byte-identical) unless the encoder
    base is trainable (``unfreeze_top_n>0`` / LoRA) AND ``optimizer.lr_encoder``
    is set: then two groups — a small-lr ``enc_opt`` over ``model.encoder.base``
    (placed first so spt's first-match-wins routing assigns the base subtree
    to it) + ``rest_opt`` over the rest — each with its own scheduler (spt
    requires #schedulers == #optimizers, module.py:309-312).
    """
    sched = {"type": "LinearWarmupCosineAnnealingLR"}
    opt_cfg = dict(cfg.optimizer)
    lr_encoder = opt_cfg.pop("lr_encoder", None)
    if lr_encoder is None or not _has_trainable_base(world_model):
        return {
            "model_opt": {
                "modules": "model", "optimizer": opt_cfg,
                "scheduler": dict(sched), "interval": "epoch",
            },
        }
    enc_cfg = dict(opt_cfg)
    enc_cfg["lr"] = lr_encoder
    return {
        "enc_opt": {
            "modules": r"model\.encoder\.base", "optimizer": enc_cfg,
            "scheduler": dict(sched), "interval": "epoch",
        },
        "rest_opt": {
            "modules": "model", "optimizer": opt_cfg,
            "scheduler": dict(sched), "interval": "epoch",
        },
    }


def run_stage0(cfg) -> None:
    train, val = _build_loaders(cfg)
    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = _stage0_optim(cfg, world_model)

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(pca_forward, cfg=cfg),
        optim=optimizers,
    )

    _launch_trainer(cfg, module, data_module)


def run_stage1(cfg) -> None:
    """Stage-1 — real feature alignment (InfoNCE + cosine), projector only.

    WM + LLM frozen. Trains the ``WorldModelProjector`` so the projected
    predicted-outcome latent lands in the LLM's representation of the
    actual result text (spec §2.4 L1). Replaces the norm-matching stub.
    """
    from pca.training import AlignStage1Module

    train, val = _build_loaders(cfg)
    world_model = _load_wm(cfg)
    projector = _build_train_projector(cfg)
    module = AlignStage1Module(
        world_model=world_model,
        projector=projector,
        llm_name=cfg.model.llm_name,
        optimizer_cfg=dict(cfg.optimizer),
        loss_cfg=dict(cfg.loss.get("align", {})),
    )
    data_module = spt.data.DataModule(train=train, val=val)
    _launch_stage_trainer(cfg, module, data_module, "stage1")


def _build_instruct_loaders(cfg) -> tuple:
    from pca.training.llava_stages import InstructDataset, collate_instruct

    val_frac = cfg.data.get("val_frac", 0.1)
    train_set = InstructDataset(cfg.data.path, "train", val_frac, cfg.seed)
    val_set = InstructDataset(cfg.data.path, "val", val_frac, cfg.seed)
    kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    gen = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(
        train_set, shuffle=True, drop_last=True,
        collate_fn=collate_instruct, generator=gen, **kwargs,
    )
    val = DataLoader(
        val_set, shuffle=False, drop_last=False,
        collate_fn=collate_instruct, **kwargs,
    )
    return train, val


def run_stage2(cfg) -> None:
    """Stage-2 — predict-conditioned instruction tuning (spec §2.4 L2).

    WM frozen except ``outcome_head`` (auxiliary verifier BCE); trains
    projector + Qwen-LoRA + outcome_head. Optionally warm-starts the
    projector from ``experiment.stage1_ckpt``.
    """
    from pca.training.llava_stages import InstructStage2Module

    world_model = _load_wm(cfg)
    _unfreeze_outcome_head(world_model)
    projector = _build_train_projector(cfg)
    stage1_ckpt = cfg.experiment.get("stage1_ckpt")
    if stage1_ckpt:
        state = torch.load(stage1_ckpt, map_location="cpu")
        projector.load_state_dict(state, strict=False)
    train, val = _build_instruct_loaders(cfg)
    module = InstructStage2Module(
        world_model=world_model,
        projector=projector,
        llm_name=cfg.model.llm_name,
        train_cfg=_stage2_train_cfg(cfg),
    )
    data_module = spt.data.DataModule(train=train, val=val)
    _launch_stage_trainer(cfg, module, data_module, "stage2")


def _unfreeze_outcome_head(world_model: torch.nn.Module) -> None:
    head = getattr(world_model, "outcome_head", None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad_(True)
    head.train()


def _stage2_train_cfg(cfg) -> dict:
    return {
        "mu": cfg.loss.get("outcome", {}).get("weight", 0.5),
        "optimizer": dict(cfg.optimizer),
        "lora": OmegaConf.to_container(cfg.model.get("lora", {}), resolve=True),
    }


# ----- stage-1/2 checkpoint saving (spec §3 R6) -----------------------


class StageCkptCallback(pl.Callback):
    """Save Stage-1 projector / Stage-2 aligned product each epoch."""

    def __init__(self, run_dir: Path, stage: str, meta: dict) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.meta = meta

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if self.stage == "stage1":
            self._save_stage1(pl_module, epoch)
        else:
            self._save_stage2(pl_module)

    def _save_stage1(self, module, epoch: int) -> None:
        d = self.run_dir / "stage1"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(
            module.projector.state_dict(), d / f"weights_epoch_{epoch}.pt"
        )

    def _save_stage2(self, module) -> None:
        d = self.run_dir / "stage2"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(module.projector.state_dict(), d / "projector.pt")
        head = getattr(module.world_model, "outcome_head", None)
        if head is not None:
            torch.save(head.state_dict(), d / "outcome_head.pt")
        module.llm.save_pretrained(str(d / "lora"))
        (d / "align_config.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )


def _stage_meta(cfg, stage: str) -> dict:
    proj = cfg.model.projector
    return {
        "stage": stage,
        "num_tokens": proj.get("num_tokens", 4),
        "cond_signal": cfg.model.get("cond_signal", "both"),
        "alpha": cfg.model.get("alpha", 0.5),
        # R5 fusion knobs (spec §3 / §2.2); calibrate_verifier overwrites.
        "alpha_pos": cfg.model.get("alpha_pos", 0.85),
        "verifier_temp": cfg.model.get("verifier_temp", 1.0),
        "w_t": cfg.model.get("w_t", 0.0),
        "llm_name": cfg.model.llm_name,
        "wm_config": cfg.model.get("wm_config_name", "wm_humaneval"),
    }


def _launch_stage_trainer(cfg, module, data_module, stage: str) -> None:
    run_dir = Path(cfg.get("output_dir", "runs/pca"), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        OmegaConf.save(cfg, fh)

    callbacks = [StageCkptCallback(run_dir, stage, _stage_meta(cfg, stage))]
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data_module)


def _launch_trainer(cfg, module, data_module) -> None:
    run_id = cfg.get("subdir") or ""
    run_dir = Path(cfg.get("output_dir", "runs/pca"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        OmegaConf.save(cfg, fh)

    callbacks = []
    if cfg.experiment.save_ckpt:
        callbacks.append(
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=1,
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data_module)


@hydra.main(
    version_base=None, config_path="./config/train", config_name="wm_stage0"
)
def run(cfg):
    if cfg.get("dry_run", False):
        print("[train_pca] dry_run=true — config resolved OK.")
        return
    stage = cfg.experiment.stage
    if stage == "stage0":
        run_stage0(cfg)
    elif stage == "stage1":
        run_stage1(cfg)
    elif stage == "stage2":
        run_stage2(cfg)
    else:
        raise ValueError(
            f"Unknown experiment.stage={stage!r} "
            "(expected stage0|stage1|stage2)"
        )


if __name__ == "__main__":
    # ensure local imports resolve when run from repo root
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run()
