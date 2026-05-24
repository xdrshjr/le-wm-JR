"""PCA training entry point (R03).

Bypasses ``swm.data.load_dataset`` by consuming ``TrajectoryDataset``
directly. Dispatches between Stage-0 (world model only) and Stage-1
(projector only) via ``cfg.experiment.stage``. Freezing is done
manually with ``requires_grad_(False)`` (R04 — Lightning 2.x has no
``Freeze`` callback).
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

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


def _build_loaders(cfg) -> tuple:
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set = _build_dataset(cfg, "train")
    val_set = _build_dataset(cfg, "val")
    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    train = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
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


def run_stage0(cfg) -> None:
    train, val = _build_loaders(cfg)
    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(pca_forward, cfg=cfg),
        optim=optimizers,
    )

    _launch_trainer(cfg, module, data_module)


def run_stage1(cfg) -> None:
    """Stage-1 — train projector only with frozen WM and frozen LLM.

    The Qwen LLM is loaded lazily inside the Lightning module so that
    Hydra config-only dry-runs do not require a HF download.
    """
    from pca.projector.mlp import (
        WorldModelProjector,
        WorldModelProjectorConfig,
    )

    train, val = _build_loaders(cfg)
    world_model = hydra.utils.instantiate(cfg.model.wm)
    if cfg.model.wm_ckpt:
        state = torch.load(cfg.model.wm_ckpt, map_location="cpu")
        world_model.load_state_dict(state, strict=False)
    _freeze(world_model)

    proj_cfg = WorldModelProjectorConfig(
        in_dim=cfg.wm.embed_dim,
        hidden_dim=cfg.model.projector.hidden_dim,
        out_dim=cfg.model.projector.out_dim,
    )
    projector = WorldModelProjector(proj_cfg)

    module = _Stage1Module(
        world_model=world_model,
        projector=projector,
        llm_name=cfg.model.llm_name,
        optimizer_cfg=dict(cfg.optimizer),
    )
    data_module = spt.data.DataModule(train=train, val=val)
    _launch_trainer(cfg, module, data_module)


class _Stage1Module(pl.LightningModule):
    """Minimal projector trainer — see Spec §2.2 Stage-1."""

    def __init__(
        self,
        world_model: torch.nn.Module,
        projector: torch.nn.Module,
        llm_name: str,
        optimizer_cfg: dict,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.projector = projector
        self.llm_name = llm_name
        self.optimizer_cfg = optimizer_cfg
        self._llm = None

    def _ensure_llm(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoModelForCausalLM

        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, torch_dtype=torch.float16
        )
        llm.gradient_checkpointing_enable()
        _freeze(llm)
        self._llm = llm.to(self.device)

    def training_step(self, batch, batch_idx):
        self._ensure_llm()
        info = self.world_model.encode(batch)
        z_t = info["emb"][:, -1]
        z_proj = self.projector(z_t)  # (B, 1, d_llm)
        # Stub Stage-1 supervision: align z_proj norm to LLM embed stats.
        embed_layer = self._llm.get_input_embeddings()
        tgt_norm = embed_layer.weight.detach().norm(dim=-1).mean()
        loss = (z_proj.norm(dim=-1) - tgt_norm).pow(2).mean()
        self.log("stage1/proj_norm_loss", loss.detach(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_cls = getattr(torch.optim, self.optimizer_cfg.pop("type"))
        return optim_cls(self.projector.parameters(), **self.optimizer_cfg)


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
    else:
        raise ValueError(
            f"Unknown experiment.stage={stage!r} (expected stage0|stage1)"
        )


if __name__ == "__main__":
    # ensure local imports resolve when run from repo root
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run()
