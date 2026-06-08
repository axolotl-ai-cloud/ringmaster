"""trl / accelerate adapter.

Bridges an ``accelerate`` run to ringmaster: pull the ``cp`` (and, for USP,
``cp_ulysses``/``cp_ring``) dims from the accelerator's device mesh and install the
CP runtime, then use ``context_parallel_region`` per step to shard the batch.

    rt = setup_from_accelerate(accelerator, RingmasterConfig(size=8), num_kv_heads=8)
    model.set_attn_implementation(rt.attn_implementation)
    for batch in loader:
        with context_parallel_region(batch) as shard:
            loss = model(**shard).loss
        loss.backward()
"""

from __future__ import annotations

from typing import Optional

from ringmaster import setup
from ringmaster.adapters.raw import context_parallel_region
from ringmaster.config import RingmasterConfig
from ringmaster.runtime import CPRuntime


def _mesh_from_accelerator(accelerator):
    for attr in ("torch_device_mesh", "device_mesh"):
        mesh = getattr(accelerator, attr, None)
        if mesh is not None:
            return mesh
    state = getattr(accelerator, "state", None)
    return getattr(state, "device_mesh", None) if state is not None else None


def setup_from_accelerate(
    accelerator,
    config: RingmasterConfig,
    *,
    num_kv_heads: Optional[int] = None,
    inner_attn: str = "flash_attention_2",
    cp_dim: str = "cp",
) -> CPRuntime:
    return setup(
        config,
        num_kv_heads=num_kv_heads,
        device_mesh=_mesh_from_accelerator(accelerator),
        cp_dim=cp_dim,
        inner_attn=inner_attn,
    )


__all__ = ["setup_from_accelerate", "context_parallel_region"]
