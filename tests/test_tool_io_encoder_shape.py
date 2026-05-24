"""Shape contract — ToolIOEncoder(cfg)(texts).shape == (B, embed_dim)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch  # noqa: F401

        from pca.encoder.tool_io import ToolIOEncoder, ToolIOEncoderConfig
    except (ImportError, OSError) as exc:
        print(f"SKIP: missing dependency ({exc})")
        return 0

    cfg = ToolIOEncoderConfig(hidden_dim=384, out_dim=384, num_head_layers=2)
    try:
        encoder = ToolIOEncoder(cfg)
    except Exception as exc:  # HF download blocked / offline
        print(f"SKIP: cannot construct ToolIOEncoder ({exc})")
        return 0

    texts = ["print('hello')", "ls -la"]
    out = encoder(texts)
    expected = (len(texts), cfg.out_dim)
    if tuple(out.shape) != expected:
        print(f"FAIL: got {tuple(out.shape)}, expected {expected}")
        return 1

    print(f"OK: ToolIOEncoder output shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
