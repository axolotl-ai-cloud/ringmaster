"""Process-wide context-parallel runtime state.

A single ``CPRuntime`` per process holds the resolved config and the Ulysses /
Ring process groups. Strategy code (e.g. the Ulysses attention function, which
is invoked deep inside the model with no access to the trainer) reads the active
runtime via ``get_runtime()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch.distributed as dist

    from ringmaster.config import RingmasterConfig

_RUNTIME: Optional["CPRuntime"] = None


@dataclass
class CPRuntime:
    config: "RingmasterConfig"
    ulysses_group: Optional["dist.ProcessGroup"] = None
    ring_group: Optional["dist.ProcessGroup"] = None
    # The full CP group (ulysses x ring flattened) — used for loss reduction and
    # input sharding, which span the entire context-parallel world.
    cp_group: Optional["dist.ProcessGroup"] = None
    # attn_implementation the host should set on the model (e.g. the registered
    # Ulysses name, or sdpa/flex for the Ring leg).
    attn_implementation: Optional[str] = None
    # Packed-sequence (varlen) metadata for the CURRENT step: (cu_seqlens int32
    # [n_seg+1] over the FULL global sequence, max_seqlen int), or None for dense.
    # Set per-step by the CP context manager; read by the Ulysses varlen path.
    varlen: Optional[tuple] = None

    @property
    def ulysses_size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(self.ulysses_group) if self.ulysses_group else 1

    @property
    def ring_size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(self.ring_group) if self.ring_group else 1

    @property
    def cp_size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(self.cp_group) if self.cp_group else 1

    @property
    def cp_rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank(self.cp_group) if self.cp_group else 0

    @property
    def shard_load_balance(self) -> str:
        """Canonical shard layout for ``shard_batch`` (single source of truth shared
        with the ring attention). Only ring + head_tail is zigzag; Ulysses, distflash
        and none all shard contiguously."""
        from ringmaster.config import LoadBalance

        if self.ring_size > 1 and self.config.load_balance == LoadBalance.HEAD_TAIL:
            return "head_tail"
        return "contiguous"


def set_runtime(runtime: Optional["CPRuntime"]) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> "CPRuntime":
    if _RUNTIME is None:
        raise RuntimeError(
            "ringmaster runtime not initialized; call ringmaster.setup() (or use a "
            "framework adapter) before the model forward."
        )
    return _RUNTIME


def maybe_runtime() -> Optional["CPRuntime"]:
    return _RUNTIME
