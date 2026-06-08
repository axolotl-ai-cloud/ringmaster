"""Raw-PyTorch / hatchery-core adapter.

A minimal context manager for users not on a trainer: set up the CP runtime, then
shard each batch and reduce the loss yourself. Mirrors the shape of
``torch.distributed.tensor.experimental.context_parallel`` but routes through
ringmaster's backend selection.
"""

from __future__ import annotations

import contextlib

from ringmaster.runtime import get_runtime, set_runtime
from ringmaster.shard import shard_batch


@contextlib.contextmanager
def context_parallel_region(batch: dict):
    """Shard ``batch`` in place for the current CP rank for the duration of the block."""
    rt = get_runtime()
    if rt.cp_size == 1:
        yield batch
        return
    sharded, _info = shard_batch(batch, cp_rank=rt.cp_rank, cp_size=rt.cp_size)
    yield sharded


__all__ = ["context_parallel_region", "get_runtime", "set_runtime"]
