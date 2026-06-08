"""Zigzag (head-tail) load-balanced ring attention with the per-step half-work skip.

Contiguous causal sharding wastes ~half the ring: rank ``r`` attends blocks
``0..r``, so the last rank does ``W``x the first rank's work. Zigzag fixes this by
splitting the sequence into ``2W`` chunks and giving rank ``r`` chunks ``r`` and
``2W-1-r`` (an early + a late chunk). With that layout the causal ring collapses to
one of three half-cost cases per step (see below), so every rank does equal work:

  * step 0          — local block, full causal (own two chunks).
  * 1 <= step <= r  — full Q  x  first half of the remote KV (its early chunk;
                      its late chunk is in the future for the local Q → skipped).
  * step > r        — second half of Q (the late chunk)  x  full remote KV
                      (the local early chunk can't attend either remote chunk).

Per-block math uses the Dao flash ops (forward returns out+lse; backward gets the
*global* merged out+lse so each block's grad uses the global softmax). dQ stays
local; dK/dV rotate around the ring (reverse) and accumulate to their owner.

Inputs are the interface layout ``[b, H, local_len, d]`` where ``local_len`` is the
two zigzag chunks concatenated (early then late). Output is ``[b, local_len, H, d]``.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from ringmaster.ring.merge import _lse_to_bshd
from ringmaster.ring.p2p_attn import _flash_ops, _peers, _prefetch, _shift_sync
from ringmaster.ring.varlen_blocks import (
    additive_doc_mask,
    doc_ids_from_cu,
    masked_block_bwd,
    masked_block_fwd,
)


def _zigzag_gidx(rank, half, world, device):
    """Global token indices of rank's two zigzag chunks [rank, 2W-1-rank], in local
    order (early chunk then late chunk)."""
    return torch.cat([
        torch.arange(rank * half, (rank + 1) * half, device=device),
        torch.arange((2 * world - 1 - rank) * half, (2 * world - rank) * half, device=device),
    ])


def _merge_full(out, lse, block_out, block_lse):
    blk_lse = _lse_to_bshd(block_lse, block_out)  # [b, s, H, 1] fp32
    blk_out = block_out.to(torch.float32)
    if out is None:
        return blk_out, blk_lse
    new_lse = torch.logaddexp(lse, blk_lse)
    out = torch.exp(lse - new_lse) * out + torch.exp(blk_lse - new_lse) * blk_out
    return out, new_lse


def _merge_slice(out, lse, block_out, block_lse, sl):
    """Merge a partial block into out[:, sl] / lse[:, sl] in place (out/lse full-size)."""
    blk_lse = _lse_to_bshd(block_lse, block_out)
    blk_out = block_out.to(torch.float32)
    o, l = out[:, sl], lse[:, sl]
    new_lse = torch.logaddexp(l, blk_lse)
    out[:, sl] = torch.exp(l - new_lse) * o + torch.exp(blk_lse - new_lse) * blk_out
    lse[:, sl] = new_lse
    return out, lse


class ZigzagRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, scaling, cu_seqlens=None):
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        send, recv = _peers(group)
        scale = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])

        qd = q.transpose(1, 2).contiguous()  # [b, L, H, d]
        cur_k = k.transpose(1, 2).contiguous()
        cur_v = v.transpose(1, 2).contiguous()
        half = qd.shape[1] // 2
        q1 = qd[:, half:].contiguous()
        ctx.group, ctx.scale, ctx.half = group, scale, half
        ctx.varlen = cu_seqlens is not None

        if ctx.varlen:
            # packed: doc-masked explicit blocks (flash varlen can't express zigzag's
            # non-contiguous chunks). Global-index masks subsume the schedule's causality.
            doc = doc_ids_from_cu(cu_seqlens, qd.device)
            g = _zigzag_gidx(rank, half, world, qd.device)
            out = lse = None
            for s in range(world):
                rem = (rank - s) % world
                if s < world - 1:
                    (nk, nv), works = _prefetch((cur_k, cur_v), group, send, recv)
                if s == 0:
                    m = additive_doc_mask(g, g, doc, True)
                    bo, bl = masked_block_fwd(qd, cur_k, cur_v, m, scale)
                    out, lse = _merge_full(out, lse, bo, bl)
                elif s <= rank:
                    kg = torch.arange(rem * half, (rem + 1) * half, device=qd.device)
                    m = additive_doc_mask(g, kg, doc, True)
                    bo, bl = masked_block_fwd(qd, cur_k[:, :half].contiguous(),
                                              cur_v[:, :half].contiguous(), m, scale)
                    out, lse = _merge_full(out, lse, bo, bl)
                else:
                    m = additive_doc_mask(g[half:], _zigzag_gidx(rem, half, world, qd.device),
                                          doc, True)
                    bo, bl = masked_block_fwd(q1, cur_k, cur_v, m, scale)
                    out, lse = _merge_slice(out, lse, bo, bl, slice(half, None))
                if s < world - 1:
                    for w in works:
                        w.wait()
                    cur_k, cur_v = nk, nv
            out_bshd = out.to(q.dtype)
            ctx.save_for_backward(qd, k.transpose(1, 2).contiguous(),
                                  v.transpose(1, 2).contiguous(), out_bshd,
                                  lse.squeeze(-1).transpose(1, 2).contiguous())
            ctx.doc = doc
            return out_bshd

        ops = _flash_ops()
        if ops is None or not q.is_cuda:
            raise RuntimeError("zigzag ring attention requires the CUDA flash kernel")
        fwd, _ = ops
        out = lse = None
        for s in range(world):
            if s < world - 1:
                (nk, nv), works = _prefetch((cur_k, cur_v), group, send, recv)
            if s == 0:
                bo, bl, _, _ = fwd(qd, cur_k, cur_v, 0.0, scale, True, -1, -1, 0.0, None, False)
                out, lse = _merge_full(out, lse, bo, bl)
            elif s <= rank:
                bo, bl, _, _ = fwd(qd, cur_k[:, :half].contiguous(), cur_v[:, :half].contiguous(),
                                   0.0, scale, False, -1, -1, 0.0, None, False)
                out, lse = _merge_full(out, lse, bo, bl)
            else:
                bo, bl, _, _ = fwd(q1, cur_k, cur_v, 0.0, scale, False, -1, -1, 0.0, None, False)
                out, lse = _merge_slice(out, lse, bo, bl, slice(half, None))
            if s < world - 1:
                for w in works:
                    w.wait()
                cur_k, cur_v = nk, nv

        out_bshd = out.to(q.dtype)  # [b, L, H, d]
        lse_bhs = lse.squeeze(-1).transpose(1, 2).contiguous()  # [b, H, L]
        ctx.save_for_backward(qd, k.transpose(1, 2).contiguous(),
                              v.transpose(1, 2).contiguous(), out_bshd, lse_bhs)
        return out_bshd

    @staticmethod
    def backward(ctx, d_out):
        qd, k_local, v_local, out, lse = ctx.saved_tensors  # [b,L,H,d]; lse [b,H,L]
        group, scale, half = ctx.group, ctx.scale, ctx.half
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        send, recv = _peers(group)

        dod = d_out.contiguous()  # [b, L, H, d]
        q1 = qd[:, half:].contiguous()
        dod1 = dod[:, half:].contiguous()
        out1 = out[:, half:].contiguous()
        lse1 = lse[:, :, half:].contiguous()  # [b, H, half]

        dq = torch.zeros_like(qd)
        cur_k, cur_v = k_local, v_local
        cur_dk = torch.zeros_like(k_local)
        cur_dv = torch.zeros_like(v_local)

        if ctx.varlen:
            doc = ctx.doc
            g = _zigzag_gidx(rank, half, world, qd.device)
            for s in range(world):
                rem = (rank - s) % world
                if s < world - 1:
                    (nk, nv), kv_works = _prefetch((cur_k, cur_v), group, send, recv)
                if s == 0:
                    m = additive_doc_mask(g, g, doc, True)
                    dqb, dkb, dvb = masked_block_bwd(dod, qd, cur_k, cur_v, out, lse, m, scale)
                    dq += dqb
                    cur_dk += dkb
                    cur_dv += dvb
                elif s <= rank:
                    kg = torch.arange(rem * half, (rem + 1) * half, device=qd.device)
                    m = additive_doc_mask(g, kg, doc, True)
                    dqb, dk0, dv0 = masked_block_bwd(
                        dod, qd, cur_k[:, :half].contiguous(), cur_v[:, :half].contiguous(),
                        out, lse, m, scale)
                    dq += dqb
                    cur_dk[:, :half] += dk0
                    cur_dv[:, :half] += dv0
                else:
                    m = additive_doc_mask(g[half:], _zigzag_gidx(rem, half, world, qd.device),
                                          doc, True)
                    dq1, dkb, dvb = masked_block_bwd(dod1, q1, cur_k, cur_v, out1, lse1, m, scale)
                    dq[:, half:] += dq1
                    cur_dk += dkb
                    cur_dv += dvb
                cur_dk = _shift_sync(cur_dk, group, send, recv)
                cur_dv = _shift_sync(cur_dv, group, send, recv)
                if s < world - 1:
                    for w in kv_works:
                        w.wait()
                    cur_k, cur_v = nk, nv
                else:
                    cur_k = _shift_sync(cur_k, group, send, recv)
                    cur_v = _shift_sync(cur_v, group, send, recv)
            return (dq.transpose(1, 2), cur_dk.transpose(1, 2), cur_dv.transpose(1, 2),
                    None, None, None)

        _, bwd = _flash_ops()
        for s in range(world):
            if s < world - 1:
                (nk, nv), kv_works = _prefetch((cur_k, cur_v), group, send, recv)
            if s == 0:
                dqb = torch.empty_like(qd)
                dkb = torch.empty_like(cur_k)
                dvb = torch.empty_like(cur_v)
                bwd(dod, qd, cur_k, cur_v, out, lse, dqb, dkb, dvb,
                    0.0, scale, True, -1, -1, 0.0, None, False, None)
                dq += dqb
                cur_dk += dkb
                cur_dv += dvb
            elif s <= rank:
                k0 = cur_k[:, :half].contiguous()
                v0 = cur_v[:, :half].contiguous()
                dqb = torch.empty_like(qd)
                dk0 = torch.empty_like(k0)
                dv0 = torch.empty_like(v0)
                bwd(dod, qd, k0, v0, out, lse, dqb, dk0, dv0,
                    0.0, scale, False, -1, -1, 0.0, None, False, None)
                dq += dqb
                cur_dk[:, :half] += dk0
                cur_dv[:, :half] += dv0
            else:
                dq1 = torch.empty_like(q1)
                dkb = torch.empty_like(cur_k)
                dvb = torch.empty_like(cur_v)
                bwd(dod1, q1, cur_k, cur_v, out1, lse1, dq1, dkb, dvb,
                    0.0, scale, False, -1, -1, 0.0, None, False, None)
                dq[:, half:] += dq1
                cur_dk += dkb
                cur_dv += dvb
            cur_dk = _shift_sync(cur_dk, group, send, recv)
            cur_dv = _shift_sync(cur_dv, group, send, recv)
            if s < world - 1:
                for w in kv_works:
                    w.wait()
                cur_k, cur_v = nk, nv
            else:
                cur_k = _shift_sync(cur_k, group, send, recv)
                cur_v = _shift_sync(cur_v, group, send, recv)
        return (dq.transpose(1, 2), cur_dk.transpose(1, 2), cur_dv.transpose(1, 2),
                None, None, None)


def zigzag_ring_attention(q, k, v, *, group, scaling, cu_seqlens=None):
    """q/k/v: [b, H, local_len, d] (two zigzag chunks). Returns [b, local_len, H, d].
    ``cu_seqlens`` set => packed (doc-masked) blocks."""
    return ZigzagRingAttention.apply(q, k, v, group, scaling, cu_seqlens)
