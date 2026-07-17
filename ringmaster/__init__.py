"""ringmaster — composable long-context attention.

Ulysses / Ring / USP sequence parallelism that wraps existing HF attention
kernels (FA2/3/4, sdpa, flex) and torch-native context parallel, with orthogonal
ALST-style memory optimizations. Framework-neutral core; thin adapters for
axolotl, trl/accelerate, and raw PyTorch.

Quick start (framework adapter does this for you)::

    import ringmaster as rm
    cfg = rm.RingmasterConfig(size=8, backend=rm.Backend.AUTO)
    runtime = rm.setup(cfg, num_kv_heads=8, device_mesh=mesh)
    model.set_attn_implementation(runtime.attn_implementation)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Optional

try:
    __version__ = _pkg_version("axolotl-ringmaster")
except PackageNotFoundError:  # source tree without installed dist metadata
    __version__ = "0.0.0"

from ringmaster.compat import has_min_torch, require_torch
from ringmaster.config import (
    AUTO,
    Backend,
    LoadBalance,
    RingImpl,
    RingmasterConfig,
    RotateMethod,
)
from ringmaster.mesh import build_runtime, intra_node_size
from ringmaster.profiling import profile_comms, profile_memory
from ringmaster.runtime import CPRuntime, get_runtime, maybe_runtime, set_runtime
from ringmaster.sp_context import ContextParallelContextManager, broadcast_batch
from ringmaster.strategies import (
    auto_select,
    is_recurrent_mixer,
    register_ulysses,
    wire_recurrent_layers,
)

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

__all__ = [
    "RingmasterConfig",
    "Backend",
    "RotateMethod",
    "LoadBalance",
    "RingImpl",
    "AUTO",
    "CPRuntime",
    "ContextParallelContextManager",
    "broadcast_batch",
    "setup",
    "teardown",
    "get_runtime",
    "maybe_runtime",
    "auto_select",
    "is_recurrent_mixer",
    "wire_recurrent_layers",
    "intra_node_size",
    "has_min_torch",
    "profile_comms",
    "profile_memory",
    "setup_from_parallelism_config",
]


def setup(
    config: RingmasterConfig,
    *,
    num_kv_heads: Optional[int] = None,
    device_mesh: "Optional[DeviceMesh]" = None,
    cp_dim: str = "cp",
    device_type: str = "cuda",
    inner_attn: str = "flash_attention_2",
) -> CPRuntime:
    """Resolve config + groups, register the backend, install the process runtime.
    ``inner_attn`` is the HF kernel the Ulysses leg wraps (flash_attention_2/3/4,
    sdpa, flex); the resulting name is exposed as ``runtime.attn_implementation``."""
    if not config.enabled:
        runtime = CPRuntime(config=config)
        runtime.attn_implementation = inner_attn  # type: ignore[attr-defined]
        set_runtime(runtime)
        return runtime

    require_torch()
    config.normalize(num_kv_heads=num_kv_heads, intra_node_size=intra_node_size())
    runtime = build_runtime(
        config, device_mesh=device_mesh, cp_dim=cp_dim, device_type=device_type
    )
    set_runtime(runtime)

    if config.backend == Backend.ULYSSES:
        runtime.attn_implementation = register_ulysses(inner_attn)  # type: ignore[attr-defined]
    elif config.backend == Backend.RING:
        from ringmaster.strategies.ring import register_ring

        runtime.attn_implementation = register_ring(  # type: ignore[attr-defined]
            config.ring_impl, inner_attn, config.rotate_method
        )
    else:
        from ringmaster.strategies.usp import register_usp

        runtime.attn_implementation = register_usp(  # type: ignore[attr-defined]
            config, inner_attn
        )

    return runtime


def setup_from_parallelism_config(parallelism_config, ring_config=None, **kwargs):
    """Install the runtime from an accelerate ParallelismConfig (see
    ``ringmaster.parallelism``)."""
    from ringmaster.parallelism import setup_from_parallelism_config as _impl

    return _impl(parallelism_config, ring_config, **kwargs)


def teardown() -> None:
    set_runtime(None)
