"""Thin wrapper — launch ``train_pca.py --config-name=projector_stage1``."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "train_pca.py"),
        "--config-name=projector_stage1",
        *sys.argv[1:],
    ]
    print("[train_projector_stage1] running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
