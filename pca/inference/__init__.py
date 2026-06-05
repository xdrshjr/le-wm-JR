"""pca.inference — lazy package init (R9 review fix).

``PCAAgent`` pulls torch at import time; loading it eagerly made every
pure-python import (``pca.inference.consensus`` — norm_repr, the interp
matrix, parse_assert_io) require a working torch, which breaks the local
Phase-0 sanity path. PEP 562 keeps ``from pca.inference import PCAAgent``
working unchanged while torch-free submodules import torch-free.
"""


def __getattr__(name):
    if name == "PCAAgent":
        from pca.inference.pca_agent import PCAAgent

        return PCAAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PCAAgent"]
