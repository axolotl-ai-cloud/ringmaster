"""Torch version gating for context parallelism.

The feature is hard-gated on torch >= 2.11 (see ``require_torch``); a version bump
only has to be reconciled here.
"""

from __future__ import annotations

from functools import lru_cache

MIN_TORCH = (2, 11)


@lru_cache(maxsize=1)
def torch_version() -> tuple[int, int]:
    import torch

    parts = torch.__version__.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


@lru_cache(maxsize=1)
def has_min_torch() -> bool:
    return torch_version() >= MIN_TORCH


def require_torch() -> None:
    if not has_min_torch():
        import torch

        major, minor = MIN_TORCH
        raise RuntimeError(
            f"ringmaster requires torch>={major}.{minor} for context parallelism; "
            f"found {torch.__version__}. Upgrade torch or disable context_parallel."
        )
