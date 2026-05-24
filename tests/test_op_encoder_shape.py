"""Shape contract — OpEncoder(cfg)(ops).shape == (B, embed_dim)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch  # noqa: F401

        from pca.action.schema import ApplyPatchArgs, RunTestArgs
        from pca.encoder.op_encoder import OpEncoder, OpEncoderConfig
    except (ImportError, OSError) as exc:
        print(f"SKIP: missing dependency ({exc})")
        return 0

    cfg = OpEncoderConfig(out_dim=384, hidden_dim=384)
    try:
        encoder = OpEncoder(cfg)
    except Exception as exc:  # tokenizer download / offline failure
        print(f"SKIP: cannot construct OpEncoder ({exc})")
        return 0

    ops = [
        ApplyPatchArgs(diff="---\n+++\n"),
        RunTestArgs(selector="tests/"),
    ]
    out = encoder(ops)
    expected = (len(ops), cfg.out_dim)
    if tuple(out.shape) != expected:
        print(f"FAIL: got {tuple(out.shape)}, expected {expected}")
        return 1

    print(f"OK: OpEncoder output shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
