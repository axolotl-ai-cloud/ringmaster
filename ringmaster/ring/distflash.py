"""DistFlashAttn / LightSeq-style load-balanced causal ring attention.

Unlike zigzag (which permutes tokens), this keeps each rank's tokens CONTIGUOUS
(rank r owns block r) and balances the causal triangle by routing work to
otherwise-idle ranks — so it composes with a sequential SSM/Mamba scan.

Schedule (P ranks, half = P//2), every causal block computed exactly once:
  * d=0          : diagonal (q_r x kv_r), causal.
  * 1<=d<=half   : PHASE-1 helper — rotate Q so rank r holds q_{r+d}; if r+d<P it
                   computes (q_{r+d} x kv_r) full and routes the partial (o,lse) back
                   to owner r+d, which merges it (online softmax).
  * half<d<P     : PHASE-2 owner — rotate KV so rank r holds kv_{r-d}; if r-d>=0 the
                   query owner r computes (q_r x kv_{r-d}) full locally.
Per-rank block counts balance (P=4 -> 3,3,2,2 vs ring's 1,2,3,4). Backward is a
hand-written mirror with explicit *ordered* collectives — distributed autograd over
collectives deadlocks because a data-dependent graph is traversed in different
orders on different ranks. Per-block math uses the Dao flash ops (GPU) or an
explicit-softmax fallback (CPU/validation); both take the GLOBAL merged out/lse so
each block's grad uses the global softmax (the ring-flash-attn identity).

q/k/v: interface [b, H, s_local, d]; returns out [b, s_local, H, d].
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from ringmaster.ring.kernels import hf_kernels_block, math_block
from ringmaster.ring.merge import update_out_and_lse
from ringmaster.ring.p2p_attn import _flash_ops, _shift_sync
from ringmaster.ring.varlen_blocks import (
    doc_ids_from_cu,
    varlen_block_bwd,
    varlen_block_fwd,
)


def _block_fwd(q, k, v, scale, causal, flash):
    """q/k/v [b,s,H,d] -> (out [b,s,H,d], lse [b,H,s])."""
    if flash:
        o, l, _, _ = _flash_ops()[0](q, k, v, 0.0, scale, causal, -1, -1, 0.0, None, False)
        return o, l
    return math_block(q, k, v, causal=causal, scaling=scale, dropout=0.0)


def _block_bwd(d_out, q, k, v, out, lse, scale, causal, flash, dq=None, dk=None, dv=None):
    """Grad of one block wrt q,k,v, using the GLOBAL out/lse. Returns dq,dk,dv [b,s,H,d].
    The flash path writes into the optional ``dq/dk/dv`` scratch buffers when given
    (lets the caller reuse one set across all non-diagonal blocks instead of a fresh
    allocation per hop)."""
    if flash:
        if dq is None:
            dq = torch.empty_like(q)
        if dk is None:
            dk = torch.empty_like(k)
        if dv is None:
            dv = torch.empty_like(v)
        _flash_ops()[1](d_out, q, k, v, out, lse, dq, dk, dv,
                        0.0, scale, causal, -1, -1, 0.0, None, False, None)
        return dq, dk, dv
    # explicit-softmax fallback (fp32), global-lse ring identity
    qf = q.transpose(1, 2).float()  # [b,H,s,d]
    kf = k.transpose(1, 2).float()
    vf = v.transpose(1, 2).float()
    dof = d_out.transpose(1, 2).float()
    of = out.transpose(1, 2).float()
    lf = lse.unsqueeze(-1)  # [b,H,s,1]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        m = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril(sk - sq)
        scores = scores.masked_fill(~m, float("-inf"))
    p = torch.exp(scores - lf)
    delta = (dof * of).sum(-1, keepdim=True)
    dv_ = torch.matmul(p.transpose(-1, -2), dof)
    dp = torch.matmul(dof, vf.transpose(-1, -2))
    ds = p * (dp - delta)
    dq_ = torch.matmul(ds, kf) * scale
    dk_ = torch.matmul(ds.transpose(-1, -2), qf) * scale
    return (dq_.transpose(1, 2).to(q.dtype), dk_.transpose(1, 2).to(k.dtype),
            dv_.transpose(1, 2).to(v.dtype))


class DistFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, scaling, cu_seqlens=None,
                attn_impl="flash_attention_2"):
        P = dist.get_world_size(group)
        r = dist.get_rank(group)
        half = P // 2
        scale = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])
        ranks = dist.get_process_group_ranks(group)
        up, down = ranks[(r + 1) % P], ranks[(r - 1) % P]
        flash = _flash_ops() is not None and q.is_cuda
        ctx.flash = flash

        qd = q.transpose(1, 2).contiguous()
        kd = k.transpose(1, 2).contiguous()
        vd = v.transpose(1, 2).contiguous()

        # packed: each block is doc-masked by its owners' global offsets (oq, ok)
        ctx.varlen = cu_seqlens is not None
        if ctx.varlen:
            doc = doc_ids_from_cu(cu_seqlens, q.device)
            Lr = qd.shape[1]
            ctx.doc, ctx.Lr, ctx.attn_impl = doc, Lr, attn_impl

            def bf(qb, kb, vb, oq, ok, diag):
                return varlen_block_fwd(qb, kb, vb, oq * Lr, ok * Lr, Lr, doc, scale,
                                        diag, attn_impl, flash)
        else:
            def bf(qb, kb, vb, oq, ok, diag):
                return _block_fwd(qb, kb, vb, scale, diag, flash)

        o, l = bf(qd, kd, vd, r, r, True)  # d=0 diagonal
        out, lse = update_out_and_lse(None, None, o, l)

        cur_k, cur_v = kd, vd  # phase 2: rotate KV, owner-local
        for d in range(1, P):
            cur_k = _shift_sync(cur_k, group, up, down)  # -> kv_{r-d}
            cur_v = _shift_sync(cur_v, group, up, down)
            if d > half and r - d >= 0:
                o, l = bf(qd, cur_k, cur_v, r, r - d, False)
                out, lse = update_out_and_lse(out, lse, o, l)

        cur_q = qd  # phase 1: rotate Q, helper computes q_{r+d} x kv_r, route partial back
        for d in range(1, half + 1):
            cur_q = _shift_sync(cur_q, group, down, up)  # -> q_{r+d}
            send_po = send_pl = None
            if r + d < P:
                send_po, send_pl = bf(cur_q, kd, vd, r + d, r, False)
            recv = _route(send_po, send_pl, group, ranks, r, d, P,
                          like_o=qd, like_l=(qd.shape[0], qd.shape[2], qd.shape[1]))
            if r - d >= 0:
                out, lse = update_out_and_lse(out, lse, recv[0], recv[1])

        ctx.save_for_backward(qd, kd, vd, out.to(q.dtype),
                              lse.squeeze(-1).transpose(1, 2).contiguous())
        ctx.group, ctx.scale, ctx.P, ctx.r, ctx.half = group, scale, P, r, half
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, d_out):
        qd, kd, vd, out, lse = ctx.saved_tensors  # [b,s,H,d]; lse [b,H,s]
        group, scale, P, r, half = ctx.group, ctx.scale, ctx.P, ctx.r, ctx.half
        flash = ctx.flash
        ranks = dist.get_process_group_ranks(group)
        up, down = ranks[(r + 1) % P], ranks[(r - 1) % P]
        dod = d_out.contiguous()
        if ctx.varlen:
            doc, Lr, attn_impl = ctx.doc, ctx.Lr, ctx.attn_impl

            def bb(dob, qb, kb, vb, ob, lb, oq, ok, diag):
                return varlen_block_bwd(dob, qb, kb, vb, ob, lb, oq * Lr, ok * Lr, Lr,
                                        doc, scale, diag, attn_impl, flash)
        else:
            # one scratch set reused across the non-diagonal block grads (no per-hop
            # alloc); the diagonal seeds the accumulators so it gets fresh buffers.
            bdq, bdk, bdv = torch.empty_like(qd), torch.empty_like(kd), torch.empty_like(vd)

            def bb(dob, qb, kb, vb, ob, lb, oq, ok, diag):
                if diag:
                    return _block_bwd(dob, qb, kb, vb, ob, lb, scale, True, flash)
                return _block_bwd(dob, qb, kb, vb, ob, lb, scale, False, flash,
                                  dq=bdq, dk=bdk, dv=bdv)

        # d=0 diagonal (local) — seeds the dq/dk/dv accumulators (kept, mutated in place)
        dq, dk, dv = bb(dod, qd, kd, vd, out, lse, r, r, True)

        # phase 2: mirror KV rotation; dk/dv for kv_{r-d} accumulate into a rotating
        # buffer that returns to the kv owner after P-1 hops (like the p2p ring bwd).
        cur_k, cur_v = kd, vd
        cur_dk = torch.zeros_like(kd)
        cur_dv = torch.zeros_like(vd)
        for d in range(1, P):
            cur_k = _shift_sync(cur_k, group, up, down)
            cur_v = _shift_sync(cur_v, group, up, down)
            if d > half and r - d >= 0:
                dq_b, dk_b, dv_b = bb(dod, qd, cur_k, cur_v, out, lse, r, r - d, False)
                dq.add_(dq_b)
                cur_dk.add_(dk_b)
                cur_dv.add_(dv_b)
            cur_dk = _shift_sync(cur_dk, group, up, down)  # rotate with KV (returns to owner)
            cur_dv = _shift_sync(cur_dv, group, up, down)
        dk.add_(cur_dk)
        dv.add_(cur_dv)

        # phase 1: owner sends (dod,q,out,lse) to its helper (r-d) and gets dq back;
        # as helper, recv owner-data from (r+d), compute its block bwd, send dq back,
        # add dk_r/dv_r locally. dn=(r-d) holds our q_r, up=(r+d) is the owner we help.
        # On gloo we post a matched send+recv on every rank (boundary ranks exchange
        # placeholder tensors the peer ignores) because asymmetric send-only/recv-only
        # ops deadlock gloo from an autograd worker thread; on NCCL the asymmetric form
        # is correct and cheaper, so we keep it (no extra boundary traffic).
        sym = dist.get_backend(group) == "gloo"
        # reusable CONTIGUOUS recv buffers (routes are synchronous → safe to reuse each
        # iteration; empty_like would inherit a saved tensor's non-contiguous strides,
        # which gloo irecv rejects — so allocate by shape).
        _buf = lambda t: torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
        r_ctx = [_buf(dod), _buf(qd), _buf(out), _buf(lse)]
        r_dq = _buf(qd)
        cur_q = qd
        for d in range(1, half + 1):
            cur_q = _shift_sync(cur_q, group, down, up)  # q_{r+d} (mirror fwd)
            is_helper = r + d < P   # we compute for owner r+d
            is_owner = r - d >= 0   # our q_r is held by helper r-d
            dn, upr = ranks[(r - d) % P], ranks[(r + d) % P]
            # send our context to our helper (dn); recv owner's context from (upr).
            rc = _bwd_route((dod, qd, out, lse), is_owner, dn, upr, is_helper, group, sym,
                            recv_into=r_ctx)
            dq_back = qd  # placeholder shape; real only when we are a valid helper
            if is_helper:
                rdod, rq, rout, rlse = rc
                dq_h, dk_h, dv_h = bb(rdod, rq, kd, vd, rout, rlse, r + d, r, False)
                dk.add_(dk_h)
                dv.add_(dv_h)
                dq_back = dq_h  # owner's dq contribution (scratch on flash; sent before reuse)
            # send dq_back to owner (upr); recv our q_r's dq from helper (dn).
            gc = _bwd_route((dq_back,), is_helper, upr, dn, is_owner, group, sym, recv_into=[r_dq])
            if is_owner:
                dq.add_(gc[0])

        return (dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2),
                None, None, None, None)


def _route(po, pl, group, ranks, r, d, P, like_o, like_l):
    """Forward partial routing: send (po,pl) to owner r+d (if <P); recv this rank's
    partial from helper r-d (if >=0). Returns (recv_o, recv_l) or (None,None)."""
    dst = ranks[r + d] if r + d < P else -1
    src = ranks[r - d] if r - d >= 0 else -1
    ops = []
    if dst >= 0:
        ops.append(dist.P2POp(dist.isend, po.contiguous(), dst, group=group))
        ops.append(dist.P2POp(dist.isend, pl.contiguous(), dst, group=group))
    ro = rl = None
    if src >= 0:
        ro = torch.empty_like(like_o)
        rl = torch.empty(like_l, dtype=torch.float32, device=like_o.device)
        ops.append(dist.P2POp(dist.irecv, ro, src, group=group))
        ops.append(dist.P2POp(dist.irecv, rl, src, group=group))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    return ro, rl


def _bwd_route(send, send_valid, send_to, recv_from, recv_valid, group, sym, recv_into=None):
    """One leg of the phase-1 backward exchange. `send` carries the tensor shapes;
    recv buffers are contiguous (gloo irecv rejects non-contiguous buffers — saved
    tensors can be non-contiguous — where NCCL only warns). `recv_into` lets the caller
    pass pre-allocated recv buffers to reuse across iterations (the routes are
    synchronous, so a buffer is never in flight when the next call reuses it). Returns
    the recv list, or None when nothing is received.

    sym=True (gloo): always post a matched send+recv so participation is symmetric;
      boundary ranks send locally-available tensors the peer ignores via its gate.
    sym=False (NCCL): post the send only when send_valid and the recv only when
      recv_valid — the cheaper asymmetric form, which NCCL handles fine."""
    def _bufs():
        if recv_into is not None:
            return recv_into
        return [torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device) for t in send]

    if sym:
        recv = _bufs()
        ops = [dist.P2POp(dist.isend, t.contiguous(), send_to, group=group) for t in send]
        ops += [dist.P2POp(dist.irecv, t, recv_from, group=group) for t in recv]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        return recv
    ops, recv = [], None
    if send_valid:
        ops += [dist.P2POp(dist.isend, t.contiguous(), send_to, group=group) for t in send]
    if recv_valid:
        recv = _bufs()
        ops += [dist.P2POp(dist.irecv, t, recv_from, group=group) for t in recv]
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    return recv


def distflash_attention(q, k, v, *, group, scaling, cu_seqlens=None,
                        attn_implementation="flash_attention_2"):
    """Balanced contiguous causal ring (DistFlashAttn-style). q/k/v: [b,H,s_local,d].
    ``cu_seqlens`` set => packed (doc-masked) blocks."""
    return DistFlashAttention.apply(q, k, v, group, scaling, cu_seqlens, attn_implementation)
