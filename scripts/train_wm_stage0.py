"""Thin wrapper — launch ``train_pca.py --config-name=wm_stage0``.

Just shells out (rather than importing) so the wrapper can be replaced
with an ``ssh_exec.py`` remote launcher without touching trainer code.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "train_pca.py"),
        "--config-name=wm_stage0",
        *sys.argv[1:],
    ]
    print("[train_wm_stage0] running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
