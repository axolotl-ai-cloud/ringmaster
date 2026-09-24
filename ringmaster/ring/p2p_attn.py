"""Memory-optimal P2P ring attention with a true flash backward (no all-gather).

O(S/P) KV resident: each rank holds its Q block, rotates K/V one hop at a time, merges
partials via online softmax. Backward is the reverse ring — rotate the dK/dV
accumulators so each block's grad returns to its owner; dQ stays local. Per-block math
uses the Dao low-level ``_flash_attn_forward``/``_backward`` with the GLOBAL merged
out/lse (the ring-flash-attn identity); falls back to explicit-softmax (O(S/P²), small
test shapes) when flash isn't importable (CPU/gloo). Interface layout ``[b,H,s_local,d]``;
causal uses contiguous sharding.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.distributed as dist

from ringmaster.ring.merge import update_out_and_lse


@lru_cache(maxsize=2)
def _flash_ops(attn_implementation: str = "flash_attention_2"):
    """Return (fwd, bwd) low-level Dao flash ops, or None if unavailable."""
    try:
        from transformers.modeling_flash_attention_utils import _lazy_imports

        fn = _lazy_imports(attn_implementation)[0]
        if fn is None:
            return None
        mod = __import__(fn.__module__, fromlist=["x"])
        return mod._flash_attn_forward, mod._flash_attn_backward
    except Exception:
        return None


def _peers(group):
    ranks = dist.get_process_group_ranks(group)
    local = dist.get_rank(group)
    n = len(ranks)
    return ranks[(local + 1) % n], ranks[(local - 1) % n]


def _ishift(t, group, send, recv):
    """Start an async ring hop; caller computes then waits the handles. Issuing the
    P2P before the block compute lets NCCL overlap the transfer with the attention
    kernels (DistFlashAttn-style comm/compute overlap)."""
    t = t.contiguous()
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    works = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, t, send, group=group),
            dist.P2POp(dist.irecv, out, recv, group=group),
        ]
    )
    return out, works


def _prefetch(tensors, group, send, recv):
    bufs, works = [], []
    for t in tensors:
        b, w = _ishift(t, group, send, recv)
        bufs.append(b)
        works.extend(w)
    return bufs, works


def _shift_sync(t, group, send, recv):
    out, works = _ishift(t, group, send, recv)
    for w in works:
        w.wait()
    return out


def _land_rot(rot):
    """Wait an in-flight (next_dk, next_dv, works) dk/dv rotation; return the buffers."""
    ndk, ndv, works = rot
    for w in works:
        w.wait()
    return ndk, ndv


# -- explicit-softmax fallback (CPU / no-flash; O(S/P^2) memory, small shapes only) --
def _block_scores(q, k, scale, block_causal):
    s = torch.matmul(q, k.transpose(-1, -2)) * scale  # [b, h, sq, sk]
    if block_causal:
        sq, sk = s.shape[-2], s.shape[-1]
        mask = torch.ones(sq, sk, dtype=torch.bool, device=s.device).tril(sk - sq)
        s = s.masked_fill(~mask, float("-inf"))
    return s


class RingP2PAttention(torch.autograd.Function):
    """Flash p2p ring. q/k/v: ``[b, H, s_local, d]``; returns out ``[b, s_local, H, d]``."""

    @staticmethod
    def forward(ctx, q, k, v, group, causal, scaling):
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        send, recv = _peers(group)
        scale = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])
        ops = _flash_ops() if q.is_cuda else None  # flash kernel is CUDA-only
        ctx.flash = ops is not None

        if ctx.flash:
            fwd, _ = ops
            qd = q.transpose(1, 2).contiguous()  # [b, s, H, d]
            cur_k = k.transpose(1, 2).contiguous()
            cur_v = v.transpose(1, 2).contiguous()
            out = lse = None
            for s in range(world):
                if s < world - 1:
                    (nk, nv), works = _prefetch((cur_k, cur_v), group, send, recv)
                if (not causal) or s <= rank:
                    block_causal = causal and s == 0
                    ob, lj, _, _ = fwd(
                        qd, cur_k, cur_v, 0.0, scale, block_causal, -1, -1, 0.0, None, False
                    )
                    out, lse = update_out_and_lse(out, lse, ob, lj)
                if s < world - 1:
                    for w in works:
                        w.wait()
                    cur_k, cur_v = nk, nv
            out_bshd = out.to(q.dtype)  # [b, s, H, d]
            lse_bhs = lse.squeeze(-1).transpose(1, 2).contiguous()  # [b, H, s] fp32
            ctx.save_for_backward(
                qd, k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
                out_bshd, lse_bhs,
            )
            ctx.group, ctx.causal, ctx.scale = group, causal, scale
            return out_bshd

        # explicit fallback
        qf = q.float()
        cur_k, cur_v = k.float().contiguous(), v.float().contiguous()
        acc = lse = None
        for s in range(world):
            if s < world - 1:
                (nk, nv), works = _prefetch((cur_k, cur_v), group, send, recv)
            if (not causal) or s <= rank:
                scores = _block_scores(qf, cur_k, scale, causal and s == 0)
                bl = torch.logsumexp(scores, dim=-1, keepdim=True)
                p = torch.exp(scores - bl)
                ob = torch.matmul(p, cur_v)
                if acc is None:
                    acc, lse = ob, bl
                else:
                    nl = torch.logaddexp(lse, bl)
                    acc = torch.exp(lse - nl) * acc + torch.exp(bl - nl) * ob
                    lse = nl
            if s < world - 1:
                for w in works:
                    w.wait()
                cur_k, cur_v = nk, nv
        ctx.save_for_backward(q, k, v, acc, lse)
        ctx.group, ctx.causal, ctx.scale = group, causal, scale
        return acc.transpose(1, 2).to(q.dtype)  # [b, s, H, d]

    @staticmethod
    def backward(ctx, d_out):
        group, causal, scale = ctx.group, ctx.causal, ctx.scale
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        send, recv = _peers(group)

        if ctx.flash:
            _, bwd = _flash_ops()
            qd, k_local, v_local, out, lse = ctx.saved_tensors  # [b,s,H,d], lse [b,H,s]
            dod = d_out.contiguous()  # [b, s, H, d]
            dq = torch.zeros_like(qd)
            cur_k, cur_v = k_local, v_local
            cur_dk = torch.zeros_like(k_local)
            cur_dv = torch.zeros_like(v_local)
            rot = None  # in-flight (next_dk, next_dv, works) dk/dv rotation
            for s in range(world):
                if s < world - 1:
                    (nk, nv), kv_works = _prefetch((cur_k, cur_v), group, send, recv)
                if (not causal) or s <= rank:
                    block_causal = causal and s == 0
                    dq_b = torch.empty_like(qd)
                    dk_b = torch.empty_like(cur_k)
                    dv_b = torch.empty_like(cur_v)
                    bwd(
                        dod, qd, cur_k, cur_v, out, lse, dq_b, dk_b, dv_b,
                        0.0, scale, block_causal, -1, -1, 0.0, None, False, None,
                    )
                    dq = dq + dq_b
                    if rot is not None:  # land prev rotation (it overlapped this compute)
                        cur_dk, cur_dv = _land_rot(rot)
                        rot = None
                    cur_dk = cur_dk + dk_b
                    cur_dv = cur_dv + dv_b
                elif rot is not None:
                    cur_dk, cur_dv = _land_rot(rot)
                    rot = None
                # rotate dk/dv one hop async; its transfer overlaps the next block's bwd
                ndk, w1 = _ishift(cur_dk, group, send, recv)
                ndv, w2 = _ishift(cur_dv, group, send, recv)
                rot = (ndk, ndv, w1 + w2)
                if s < world - 1:
                    for w in kv_works:
                        w.wait()
                    cur_k, cur_v = nk, nv
                else:
                    cur_k = _shift_sync(cur_k, group, send, recv)
                    cur_v = _shift_sync(cur_v, group, send, recv)
            if rot is not None:
                cur_dk, cur_dv = _land_rot(rot)
            # [b, s, H, d] -> interface grad layout [b, H, s, d]
            return (
                dq.transpose(1, 2), cur_dk.transpose(1, 2), cur_dv.transpose(1, 2),
                None, None, None,
            )

        # explicit fallback
        q, k, v, out, lse = ctx.saved_tensors
        qf, dof = q.float(), d_out.transpose(1, 2).float()  # d_out [b,s,H,d] -> [b,H,s,d]
        delta = (dof * out).sum(dim=-1, keepdim=True)
        dq = torch.zeros_like(qf)
        cur_k, cur_v = k.float(), v.float()
        cur_dk, cur_dv = torch.zeros_like(cur_k), torch.zeros_like(cur_v)
        rot = None  # in-flight (next_dk, next_dv, works) dk/dv rotation
        for s in range(world):
            if s < world - 1:
                (nk, nv), kv_works = _prefetch((cur_k, cur_v), group, send, recv)
            if (not causal) or s <= rank:
                scores = _block_scores(qf, cur_k, scale, causal and s == 0)
                p = torch.exp(scores - lse)
                dv = torch.matmul(p.transpose(-1, -2), dof)
                dp = torch.matmul(dof, cur_v.transpose(-1, -2))
                ds = p * (dp - delta)
                dq = dq + torch.matmul(ds, cur_k) * scale
                if rot is not None:  # land prev rotation (it overlapped this compute)
                    cur_dk, cur_dv = _land_rot(rot)
                    rot = None
                cur_dk = cur_dk + torch.matmul(ds.transpose(-1, -2), qf) * scale
                cur_dv = cur_dv + dv
            elif rot is not None:
                cur_dk, cur_dv = _land_rot(rot)
                rot = None
            ndk, w1 = _ishift(cur_dk, group, send, recv)
            ndv, w2 = _ishift(cur_dv, group, send, recv)
            rot = (ndk, ndv, w1 + w2)
            if s < world - 1:
                for w in kv_works:
                    w.wait()
                cur_k, cur_v = nk, nv
            else:
                cur_k = _shift_sync(cur_k, group, send, recv)
                cur_v = _shift_sync(cur_v, group, send, recv)
        if rot is not None:
            cur_dk, cur_dv = _land_rot(rot)
        return dq.to(q.dtype), cur_dk.to(k.dtype), cur_dv.to(v.dtype), None, None, None


def ring_p2p_attention(q, k, v, *, group, causal, scaling):
    """q/k/v: [b, H, s_local, d]. Returns [b, s_local, H, d] (interface output layout)."""
    if q.shape[1] != k.shape[1] and (not q.is_cuda or _flash_ops() is None):
        if q.shape[1] % k.shape[1]:
            raise ValueError("Query heads must be divisible by KV heads")
        repeats = q.shape[1] // k.shape[1]
        k, v = (t.repeat_interleave(repeats, dim=1) for t in (k, v))
    return RingP2PAttention.apply(q, k, v, group, causal, scaling)
