"""Compatibility with accelerate's ParallelismConfig (which axolotl uses).

Consumes the device mesh ParallelismConfig builds and reads its ``cp`` dim, so a host
configured with ``ParallelismConfig(cp_size=...)`` composes directly. For USP the
``cp`` dim may be split into ``cp_ulysses``/``cp_ring`` at construction; the mesh
builder accepts either form.
"""

from __future__ import annotations

from typing import Optional

from ringmaster import setup
from ringmaster.config import RingmasterConfig
from ringmaster.runtime import CPRuntime


def _device_mesh(pc, device_type: str):
    getter = getattr(pc, "get_device_mesh", None)
    if callable(getter):
        try:
            mesh = getter(device_type)
            if mesh is not None:
                return mesh
        except Exception:  # pragma: no cover - some versions require build first
            pass
    builder = getattr(pc, "build_device_mesh", None)
    if callable(builder):
        return builder(device_type)
    return None


def setup_from_parallelism_config(
    parallelism_config,
    ring_config: Optional[RingmasterConfig] = None,
    *,
    num_kv_heads: Optional[int] = None,
    inner_attn: str = "flash_attention_2",
    device_type: str = "cuda",
    cp_dim: str = "cp",
) -> CPRuntime:
    """Install the ringmaster runtime from an accelerate ParallelismConfig.

    ``ring_config`` lets you choose backend/ring_impl/etc.; its ``size`` defaults to
    the ParallelismConfig's ``cp_size``.
    """
    cp_size = int(getattr(parallelism_config, "cp_size", 1) or 1)
    if ring_config is None:
        ring_config = RingmasterConfig(size=cp_size)
    elif ring_config.size <= 1:
        ring_config.size = cp_size

    mesh = _device_mesh(parallelism_config, device_type)
    return setup(
        ring_config,
        num_kv_heads=num_kv_heads,
        device_mesh=mesh,
        cp_dim=cp_dim,
        device_type=device_type,
        inner_attn=inner_attn,
    )


__all__ = ["setup_from_parallelism_config"]
