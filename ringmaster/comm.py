"""Communication primitives for sequence parallelism.

The Ulysses leg is a pair of all-to-alls around the attention call: one to turn a
sequence-sharded tensor ``[b, H, s/P, d]`` into a head-sharded ``[b, H/P, s, d]``
and one to turn the result back. ``SeqAllToAll4D`` is the autograd-aware version;
its backward is the same all-to-all with scatter/gather swapped.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ringmaster.profiling import timed_collective


def _exchange(inputs: list[torch.Tensor], group: dist.ProcessGroup) -> list[torch.Tensor]:
    """all_to_all over a list of equal-shaped tensors (gloo lacks it → all_gather
    emulation, so the same path runs on CPU for tests)."""
    world = len(inputs)
    # collectives need contiguous buffers and empty_like inherits a view's strides,
    # so allocate outputs from the contiguous inputs.
    inputs = [t.contiguous() for t in inputs]
    nbytes = sum(t.numel() * t.element_size() for t in inputs)
    if dist.get_backend(group) == "gloo":
        stacked = torch.stack(inputs, dim=0)
        gathered = [torch.empty_like(stacked) for _ in range(world)]
        timed_collective(
            "all_to_all(gloo)", nbytes, lambda: dist.all_gather(gathered, stacked, group=group)
        )
        rank = dist.get_rank(group)
        return [gathered[src][rank].contiguous() for src in range(world)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(world)]
    timed_collective(
        "all_to_all", nbytes, lambda: dist.all_to_all(outputs, inputs, group=group)
    )
    return outputs


def _all_to_all(
    x: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Split ``x`` on ``scatter_dim`` into P, exchange, concat on ``gather_dim``
    (``size(scatter_dim)`` must be divisible by P). NCCL uses one fused
    ``all_to_all_single`` over a contiguous buffer; gloo keeps the all_gather emulation.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if dist.get_backend(group) == "gloo":
        outputs = _exchange(list(x.chunk(world, dim=scatter_dim)), group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    sd, gd = scatter_dim % x.dim(), gather_dim % x.dim()
    s = x.shape[sd]
    assert s % world == 0, f"scatter dim {sd}={s} not divisible by world {world}"
    # [.. S ..] -> [world, S/world, *rest] with dim 0 enumerating the scatter chunks
    t = x.movedim(sd, 0).contiguous()
    inp = t.reshape(world, s // world, *t.shape[1:])
    out = torch.empty_like(inp)
    nbytes = inp.numel() * inp.element_size()
    timed_collective("all_to_all", nbytes, lambda: dist.all_to_all_single(out, inp, group=group))
    # out[i] = source i's scatter-chunk for this rank; concat the P source slices along
    # gather_dim: scatter axis back to its slot, then merge the source axis into gather.
    out = out.movedim(1, sd + 1).movedim(0, gd)
    shp = out.shape
    return out.contiguous().reshape(*shp[:gd], world * shp[gd + 1], *shp[gd + 2:])


class SeqAllToAll4D(torch.autograd.Function):
    """Autograd-aware all-to-all over a 4D tensor for Ulysses attention."""

    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad):
        # The inverse of (scatter, gather) is (gather, scatter).
        out = _all_to_all(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group)
        return None, out, None, None


def seq_all_to_all(
    x: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    return SeqAllToAll4D.apply(group, x, scatter_dim, gather_dim)
