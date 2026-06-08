"""Autograd-aware CP gathers.

``AllGatherCP`` (recurrent state pass): backward SUMs the gathered grads back to
each owner — the carry state is shared across ranks. ``SeqGatherCP`` (GRPO/EBFT
output gather): each rank owns a disjoint seq slice, so backward just slices out
this rank's chunk — no cross-rank sum.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class AllGatherCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        world = dist.get_world_size(group)
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(out, x, group=group)
        return torch.stack(out, dim=0)

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        rank = dist.get_rank(group)
        g = grad.contiguous().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=group)
        return g[rank], None


def all_gather_cp(x, group):
    return AllGatherCP.apply(x, group)


class SeqGatherCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world = dist.get_world_size(group)
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(ctx.world)]
        dist.all_gather(out, x, group=group)
        # rank-ordered slices -> contiguous full sequence (dim 1)
        return torch.cat(out, dim=1)

    @staticmethod
    def backward(ctx, grad):
        # each rank owns a disjoint slice: take this rank's chunk, no all_reduce
        return grad.contiguous().chunk(ctx.world, dim=1)[ctx.rank].contiguous(), None


def seq_gather_cp(x, group):
    return SeqGatherCP.apply(x, group)
