"""Framework-neutral configuration for ringmaster.

Frameworks (axolotl, trl, raw torch) translate their own config into a
``RingmasterConfig``. The core never imports pydantic or any framework so it can
be vendored or pip-installed standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    AUTO = "auto"
    ULYSSES = "ulysses"
    RING = "ring"
    USP = "usp"


class RotateMethod(str, Enum):
    ALLGATHER = "allgather"
    ALLTOALL = "alltoall"


class LoadBalance(str, Enum):
    NONE = "none"  # contiguous shard; causal ring is imbalanced (fastest, simplest)
    HEAD_TAIL = "head_tail"  # zigzag: reorder to chunks [r, 2W-1-r]; balanced but
    # permutes tokens → unsafe with a sequential SSM/Mamba scan (attention-only).
    DISTFLASH = "distflash"  # DistFlashAttn-style: contiguous (no permute → SSM-safe)
    # + balanced by routing work to idle ranks. Ring-only.
    PER_DOCUMENT = "per_document"  # v2 (packing/varlen)
    PTRR = "ptrr"  # v2 (FlexAttention irregular masks)


class RingImpl(str, Enum):
    """Block-kernel provider for the Ring leg.

    The ring loop (KV rotation + online-softmax LSE merge + zigzag) is ours; only
    the per-block attention kernel differs, so Ring is *not* limited to torch's
    native SDPA/flex.
    """

    AUTO = "auto"  # hf_kernels when the model attn is a flash kernel, else torch_native
    TORCH_NATIVE = "torch_native"  # aten ring SDPA/flex (merge inside aten)
    HF_KERNELS = "hf_kernels"  # our ring loop over HF-kernels flash (FA2/3/4), returns LSE
    RING_FLASH_ATTN = "ring_flash_attn"  # legacy flash_attn pypi, opt-in


# Sentinel for "let the auto-selector decide".
AUTO = -1


@dataclass
class RingmasterConfig:
    size: int = 1
    backend: Backend = Backend.AUTO
    ulysses_size: int = AUTO
    ring_size: int = AUTO
    rotate_method: RotateMethod = RotateMethod.ALLGATHER
    # NONE (contiguous) is the safe default: always correct, SSM-safe, and matches
    # shard_batch's contiguous default. head_tail (zigzag) reorders tokens and needs
    # matching zigzag sharding, so it (and distflash) are explicit opt-in.
    load_balance: LoadBalance = LoadBalance.NONE
    ring_impl: RingImpl = RingImpl.AUTO

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def normalize(self, *, num_kv_heads: int | None, intra_node_size: int | None):
        """Resolve AUTO fields into concrete ulysses/ring sizes and backend.

        Pure here (no distributed calls) so it is unit-testable; the real
        topology probe in ``mesh.py`` supplies ``intra_node_size``.
        """
        from ringmaster.strategies.usp import auto_select

        ulysses, ring, backend = auto_select(
            total=self.size,
            requested_backend=self.backend,
            requested_ulysses=self.ulysses_size,
            requested_ring=self.ring_size,
            num_kv_heads=num_kv_heads,
            intra_node_size=intra_node_size,
        )
        self.ulysses_size = ulysses
        self.ring_size = ring
        self.backend = backend
        return self
