"""Compute val pred_loss for a TextJEPA, with optional checkpoint loading.

Two invocation modes:

  baseline (no ckpt, fresh random init):
      python scripts/eval_wm.py --config-name=wm_stage0_mvp split=val

  trained:
      python scripts/eval_wm.py --config-name=wm_stage0_mvp split=val \\
          ckpt=/path/to/weights_epoch_6.pt

Writes ``runs/pca_mvp/eval_<split>_<tag>.json`` and prints a short
summary that downstream scripts can grep.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

# Allow ``python scripts/eval_wm.py`` from le-wm-JR root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pca.data.collate import collate_trajectories  # noqa: E402
from pca.data.dataset import (  # noqa: E402
    TrajectoryDataset,
    TrajectoryDatasetConfig,
)


def _build_loader(cfg, split: str) -> DataLoader:
    ds_cfg = TrajectoryDatasetConfig(
        path=cfg.data.path,
        split=split,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        seed=cfg.seed,
        max_obs_chars=cfg.data.get("max_obs_chars", 16000),
    )
    ds = TrajectoryDataset(ds_cfg)
    return DataLoader(
        ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_trajectories,
        num_workers=0,
    )


def _eval_pred_loss(model, loader, ctx_len: int, n_preds: int, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            info = model.encode(batch)
            emb = info["emb"].to(device)
            act_emb = info["act_emb"].to(device)
            ctx_emb = emb[:, :ctx_len]
            ctx_act = act_emb[:, :ctx_len]
            tgt_emb = emb[:, n_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)
            mse = (pred_emb - tgt_emb).pow(2).mean().item()
            total += mse * emb.size(0)
            count += emb.size(0)
    return total / max(count, 1)


@hydra.main(
    version_base=None,
    config_path="../config/train",
    config_name="wm_stage0_mvp",
)
def run(cfg):
    split = cfg.get("split", "val")
    ckpt = cfg.get("ckpt", None)
    tag = cfg.get("tag", "trained" if ckpt else "baseline_random_init")

    model = hydra.utils.instantiate(cfg.model)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[eval_wm] loaded {ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print("[eval_wm] baseline mode — random init, no checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = _build_loader(cfg, split)
    pred_loss = _eval_pred_loss(
        model,
        loader,
        ctx_len=cfg.wm.history_size,
        n_preds=cfg.wm.num_preds,
        device=device,
    )

    out_dir = Path(cfg.get("output_dir", "runs/pca_mvp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{split}_{tag}.json"
    payload = {
        "split": split,
        "tag": tag,
        "ckpt": ckpt,
        "pred_loss_mean": pred_loss,
        "data_path": cfg.data.path,
        "embed_dim": cfg.wm.embed_dim,
        "history_size": cfg.wm.history_size,
        "num_preds": cfg.wm.num_preds,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[eval_wm] split={split} tag={tag} "
        f"pred_loss_mean={pred_loss:.6f} -> {out}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run()
