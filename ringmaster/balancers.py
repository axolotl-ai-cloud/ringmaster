"""Sequence load-balancing index generators (Ring leg only — Ulysses needs none).

Contiguous sharding under a causal mask leaves early-token ranks idle; ``head_tail``
(zigzag) pairs an early chunk with a late chunk per rank to flatten the triangle.
Pure index functions so they're CPU-unit-testable.
"""

from __future__ import annotations


def contiguous_indices(seq_len: int, world: int) -> list[list[int]]:
    if seq_len % world != 0:
        raise ValueError(f"seq_len {seq_len} not divisible by world {world}")
    chunk = seq_len // world
    return [list(range(r * chunk, (r + 1) * chunk)) for r in range(world)]


def head_tail_indices(seq_len: int, world: int) -> list[list[int]]:
    """Zigzag: rank r holds chunk r and chunk (2*world-1-r) of 2*world chunks."""
    if seq_len % (2 * world) != 0:
        raise ValueError(
            f"seq_len {seq_len} not divisible by 2*world ({2 * world}) for head_tail"
        )
    chunk = seq_len // (2 * world)

    def block(i: int) -> list[int]:
        return list(range(i * chunk, (i + 1) * chunk))

    return [block(r) + block(2 * world - 1 - r) for r in range(world)]
