"""Lightweight comm + memory profiling. All collectives route through
``timed_collective``, so ``profile_comms()`` records per-op count/bytes/time
(zero overhead when inactive — one ``is None`` check). ``profile_memory()`` tracks
peak allocation.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

_ACTIVE: Optional["CommStats"] = None


@dataclass
class CommStats:
    count: dict[str, int] = field(default_factory=dict)
    nbytes: dict[str, int] = field(default_factory=dict)
    ms: dict[str, float] = field(default_factory=dict)

    def record(self, op: str, nbytes: int, ms: float) -> None:
        self.count[op] = self.count.get(op, 0) + 1
        self.nbytes[op] = self.nbytes.get(op, 0) + nbytes
        self.ms[op] = self.ms.get(op, 0.0) + ms

    @property
    def total_bytes(self) -> int:
        return sum(self.nbytes.values())

    @property
    def total_ms(self) -> float:
        return sum(self.ms.values())

    def report(self) -> str:
        lines = ["comm profile (op: count, MiB, ms):"]
        for op in sorted(self.count):
            lines.append(
                f"  {op}: {self.count[op]}x, "
                f"{self.nbytes[op] / 2**20:.1f} MiB, {self.ms[op]:.2f} ms"
            )
        lines.append(
            f"  TOTAL: {self.total_bytes / 2**20:.1f} MiB, {self.total_ms:.2f} ms"
        )
        return "\n".join(lines)


@contextlib.contextmanager
def profile_comms():
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, CommStats()
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = prev


def timed_collective(op: str, nbytes: int, fn: Callable[[], object]):
    """Run ``fn`` (a collective), recording bytes + time if a profiler is active."""
    if _ACTIVE is None:
        return fn()
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        _ACTIVE.record(op, nbytes, start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        result = fn()
        _ACTIVE.record(op, nbytes, (time.perf_counter() - t0) * 1e3)
    return result


@dataclass
class MemStats:
    peak_mb: float = 0.0

    def report(self) -> str:
        return f"memory profile: peak {self.peak_mb:.1f} MiB allocated"


@contextlib.contextmanager
def profile_memory(device: Optional[int] = None):
    stats = MemStats()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        try:
            yield stats
        finally:
            stats.peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
    else:
        yield stats
