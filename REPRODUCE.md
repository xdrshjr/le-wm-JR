# Two-Room 复现指南（le-wm-JR fork）

本仓库是 [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) 在 commit `c8a4417` 的 fork，
附带一次在 4×RTX 2080 Ti 上完整跑通 Two-Room 实验所需要的**全部依赖代码、补丁与冻结依赖清单**，
方便在任意机器上一次性拉取并复现。

---

## 1. 仓库布局

```
le-wm-JR/
├── train.py / eval.py / jepa.py / module.py / utils.py / config/  # 上游 le-wm 原文件 @ c8a4417
├── external/
│   ├── stable-pretraining/      # main 分支 @ bce7c8b，已应用 torchvision shim
│   └── stable-worldmodel/       # main 分支 @ 64673b0，未改动
├── patches/
│   └── spt-torchvision-shim.patch  # 上面 shim 的可复用 patch（基于 bce7c8b）
├── requirements-old.txt         # 复现成功时的 pip freeze（torch 2.5.1+cu121）
├── requirements_frozen.txt      # 上游 README 推荐的更早 pin（torch 2.4.1+cu121）— 仅供对比
├── conda-env-old.yml            # 复现成功时的 conda env export
├── environment.json             # 远端 Python / GPU / CUDA 探测产物
└── REPRODUCE.md                 # 本文件
```

> 上游 README（`README.md`）保持原样未改，仅本文件描述 fork 增量。

## 2. 复现环境（实测，2026-05-20）

| 项目 | 值 |
|---|---|
| OS | Linux x86_64（kernel `5.x`） |
| Python | 3.10.20（conda env 名 `lewm`） |
| GPU | 4× RTX 2080 Ti (Turing, 22 GiB)，CUDA 12.1，驱动 `>=525` |
| PyTorch | `torch==2.5.1+cu121`、`torchvision==0.20.1+cu121` |
| 关键依赖 | `transformers==4.55.4`、`lightning>=2.x`、`hydra-core`、`lancedb`、`pylance`、`loguru`、`tabulate`、`einops`、`opencv-python-headless`、`paramiko`（仅复现脚手架需要） |
| 数据集 | `quentinll/lewm-tworooms` 的 `tworoom.tar.zst`（3.4 GiB 压缩 / 12 GiB 解压），落到 `~/.stable-wm/datasets/tworoom.h5` |

⚠️ **重要偏离上游 README**：上游 README 例子 `python train.py data=tworoom` 直接跑会触发
`AttributeError: module 'stable_worldmodel.data' has no attribute 'load_dataset'`，
因为 PyPI 上的 `stable-worldmodel==0.0.6` 落后于 main 分支。本 fork 已经把 main 分支 SWM/SPT
vendor 到 `external/`，按下面步骤装即可。

## 3. 端到端复现步骤

```bash
# 0. 克隆 fork
git clone https://github.com/xdrshjr/le-wm-JR.git
cd le-wm-JR

# 1. 创建 conda 环境（任选其一）
#    1a. 从 conda yml 一键拉起
conda env create -n lewm -f conda-env-old.yml
#    1b. 或者手动: conda create -n lewm python=3.10 -y && conda activate lewm

conda activate lewm

# 2. 安装 PyTorch 2.5.1 + cu121（先于其他包，避免 torch 被覆盖）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 vendor 的 stable-worldmodel / stable-pretraining（main 分支版本）
pip install -e external/stable-worldmodel[train,env] --no-deps
pip install -e external/stable-pretraining --no-deps

# 4. 补齐其余 pin（与服务器实测一致）
pip install -r requirements-old.txt --no-deps    # 已包含上面两个 -e 的 pin

# 5. 下载数据集（推荐 hf-mirror.com 镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download quentinll/lewm-tworooms tworoom.tar.zst --repo-type dataset \
    --local-dir ~/.stable-wm/datasets/_tmp/
mkdir -p ~/.stable-wm/datasets
zstd -d ~/.stable-wm/datasets/_tmp/tworoom.tar.zst -o ~/.stable-wm/datasets/_tmp/tworoom.tar
tar -xf ~/.stable-wm/datasets/_tmp/tworoom.tar -C ~/.stable-wm/datasets/
rm -rf ~/.stable-wm/datasets/_tmp

# 6. 训练（4 卡，约 ~63 min × 3 epoch / ~21 min per epoch）
python train.py data=tworoom seed=0 trainer.max_epochs=3 \
    trainer.devices=4 trainer.precision=16-mixed \
    2>&1 | tee runs/train_tworoom_seed0_4gpu_3ep.log
# checkpoint 落到 ~/.stable-wm/checkpoints/lewm/weights_epoch_{1,2,3}.pt

# 7. 评估（用本地 checkpoint，绕开 HF 直连）
python eval.py --config-name=tworoom.yaml policy=lewm/weights_epoch_3.pt \
    2>&1 | tee runs/eval_tworoom.log
```

## 4. SPT 兼容性补丁说明

`external/stable-pretraining/stable_pretraining/data/transforms.py` 在 import 段加了一段 monkey-patch：

```python
if not hasattr(v2.Transform, "transform"):
    def _sp_transform_proxy(self, *args, **kwargs):
        return self._transform(*args, **kwargs)
    v2.Transform.transform = _sp_transform_proxy
```

**原因**：stable-pretraining main 分支假定 `torchvision >= 0.21` 暴露的 `v2.Transform.transform()`，
但本仓库实测用的是 `torchvision==0.20.1+cu121`（Turing GPU 配对最稳定的版本，CUDA 12.1），
该版本只有 `_transform()`，调用 `transform()` 直接 `AttributeError`。shim 安装一个代理方法，
子类对 `_transform` 的 override 会被正确转发。

如果你想把 patch 重应用到 upstream SPT@bce7c8b：

```bash
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
git checkout bce7c8b35a62399d0529068ed7fea5dd2ce9021e
git apply ../patches/spt-torchvision-shim.patch
```

如果你升级到 `torchvision>=0.21`（且确认 `v2.Transform.transform` 已经存在），
shim 的 `if not hasattr(...)` 守卫会自动让 patch 变成 no-op，不需要回滚。

## 5. 已实测指标（仅供对照，2026-05-20 在 4×RTX 2080 Ti，3 epoch）

| 指标 | 值 |
|---|---|
| 训练总耗时 | 3811 s ≈ **63.5 min** |
| `fit/loss` (epoch 1→3) | 0.712 → 0.622 → **0.586** |
| `pred_loss` (epoch 3) | **0.233** |
| `sigreg_loss` (epoch 3) | **3.92** |
| `validate/loss` (epoch 3) | **0.623** |
| `eval` rollout success | **82.0 % (41/50 episodes)** |
| 模型参数量 | 18.034 M |

上游 LeWM 在 Two-Room 论文 Table 报约 90+%；3 epoch 训练已能到 82%。
跑满上游默认 `max_epochs=100` 预期可以贴近论文数值。

## 6. 已知坑

- **stable-worldmodel PyPI `0.0.6` 缺 `swm.data.load_dataset`** → 必须用 main 分支（已 vendor）。
- **HuggingFace 直连超时**（CN 防火墙）→ 评估时如果用 `policy=tworoom/lewm` 会拉 HF，
  改用 `policy=lewm/weights_epoch_3.pt` 直接读本地 checkpoint。
- **RTX 2080 Ti 无 bf16** → 必须 `trainer.precision=16-mixed`，不能用上游默认的 `bf16-mixed`。
- **`lr_scheduler.step()` 早于 `optimizer.step()` 告警** → SPT main 分支已知行为，
  仅丢第 1 步 LR，对最终曲线影响 < 1 %，可忽略。

## 7. 上游引用

- LeWM: <https://github.com/lucas-maes/le-wm> @ `c8a4417`（本仓库主体 git history 保留）
- stable-worldmodel: <https://github.com/galilai-group/stable-worldmodel> @ `64673b0bceffb3787c5fb1bdc17a4ebb2c9bcc9b`
- stable-pretraining: <https://github.com/galilai-group/stable-pretraining> @ `bce7c8b35a62399d0529068ed7fea5dd2ce9021e`
