"""The ring attention loop. Inputs are interface layout ``[b, H, s_local, d]``; each
query block attends to causal KV blocks ``0..rank`` (diagonal block causal), merged
with online softmax. Per-block kernel pluggable (``ring.kernels.PROVIDERS``).

  * allgather: gather full K/V then a local pass — more memory, fewer collectives.
  * p2p: rotate KV one hop (O(S/P) resident), delegated to ``ring.p2p_attn``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ringmaster.config import RotateMethod
from ringmaster.profiling import timed_collective
from ringmaster.ring.kernels import PROVIDERS
from ringmaster.ring.merge import update_out_and_lse


def ring_attention(
    q,
    k,
    v,
    *,
    group,
    causal: bool,
    scaling,
    dropout: float,
    provider: str,
    rotate_method: RotateMethod,
    attn_implementation: str,
    window=None,
):
    block = PROVIDERS[provider]
    world = dist.get_world_size(group) if group is not None else 1
    if world == 1:
        out, _ = block(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=causal,
            scaling=scaling,
            dropout=dropout,
            attn_implementation=attn_implementation,
            window=window,
        )
        return out
    if rotate_method == RotateMethod.ALLGATHER:
        return _allgather_ring(
            q, k, v, group, causal, scaling, dropout, block, attn_implementation, window
        )
    # P2P ring: O(S/P) KV resident, trains via a flash-style reverse-ring backward.
    from ringmaster.ring.p2p_attn import ring_p2p_attention

    return ring_p2p_attention(q, k, v, group=group, causal=causal, scaling=scaling)


def _causal_blocks(rank: int, world: int, causal: bool):
    return range(rank + 1) if causal else range(world)


class _AllGatherKV(torch.autograd.Function):
    """all_gather whose backward reduce-scatters grads back to each KV owner: a key
    block owned by rank r is attended by all ranks i>=r, so its grad is their sum.
    torch's autograd all_gather doesn't do this."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        world = dist.get_world_size(group)
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(out, x, group=group)
        return torch.stack(out, dim=0)  # [P, b, s_local, h, d]

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if dist.get_backend(group) == "gloo":
            g = grad.contiguous().clone()
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=group)
            return g[rank], None
        grad_in = torch.empty_like(grad[0])
        dist.reduce_scatter(
            grad_in, [grad[i].contiguous() for i in range(world)],
            op=dist.ReduceOp.SUM, group=group,
        )
        return grad_in, None


def _allgather_ring(q, k, v, group, causal, scaling, dropout, block, attn_impl, window):
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    qb = q.transpose(1, 2).contiguous()  # [b, s_local, h, d]
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()

    nbytes = kb.numel() * kb.element_size()
    k_blocks = timed_collective(
        "all_gather", nbytes, lambda: list(_AllGatherKV.apply(kb, group).unbind(0))
    )
    v_blocks = timed_collective(
        "all_gather", nbytes, lambda: list(_AllGatherKV.apply(vb, group).unbind(0))
    )

    out = lse = None
    for j in _causal_blocks(rank, world, causal):
        bo, bl = block(
            qb,
            k_blocks[j],
            v_blocks[j],
            causal=causal and (j == rank),
            scaling=scaling,
            dropout=dropout,
            attn_implementation=attn_impl,
            window=window,
        )
        out, lse = update_out_and_lse(out, lse, bo, bl)
    return out.to(qb.dtype)
