"""Document-masked attention block (explicit softmax) for packed-sequence ring CP.

The flash block kernels only express causal/non-causal, so the load-balanced ring
backends (zigzag, distflash) use these for varlen: an additive document mask
(same-document AND global-causal) is applied per block, and the backward takes the
GLOBAL merged out/lse (the ring-flash-attn identity) so it composes with the
hand-written reverse-ring backward exactly as the flash path does. fp32 internally;
O(Lq*Lk) per block (the price of an arbitrary mask without a custom flash kernel).

Layout: q/k/v ``[b, s, H, d]`` (GQA via head repeat), out ``[b, sq, H, d]``,
lse ``[b, H, sq]``.
"""

from __future__ import annotations

from functools import lru_cache

import torch

NEG_INF = float("-inf")


@lru_cache(maxsize=2)
def _flash_varlen_ops(attn_implementation: str = "flash_attention_2"):
    """Low-level Dao (fwd, bwd) varlen ops, or None (CPU/no-flash)."""
    from transformers.modeling_flash_attention_utils import _lazy_imports

    fn = _lazy_imports(attn_implementation)[0]
    if fn is None:
        return None
    import importlib

    mod = importlib.import_module(fn.__module__)
    return mod._flash_attn_varlen_forward, mod._flash_attn_varlen_backward


def _seg_cu(doc_ids_range):
    """cu_seqlens over a contiguous range whose doc ids are non-decreasing: boundaries
    where the doc id changes, plus the endpoints. Returns (cu int32, max_seglen int)."""
    L = doc_ids_range.shape[0]
    dev = doc_ids_range.device
    change = torch.ones(L + 1, dtype=torch.bool, device=dev)
    change[1:L] = doc_ids_range[1:] != doc_ids_range[:-1]
    cu = torch.nonzero(change, as_tuple=False).view(-1).to(torch.int32)
    return cu, int(cu.diff().max().item())


def doc_ids_from_cu(cu_seqlens, device):
    """Per-token document id over the full sequence, from flash cu_seqlens."""
    seg = cu_seqlens.to(device).diff()
    return torch.repeat_interleave(torch.arange(seg.numel(), device=device), seg)


def additive_doc_mask(q_gidx, k_gidx, doc_ids, causal: bool):
    """Additive [Lq, Lk] mask: 0 where q and k share a document (and, if ``causal``,
    k is at/-before q in global order), else -inf. ``*_gidx`` are global token indices."""
    allow = doc_ids[q_gidx].unsqueeze(1) == doc_ids[k_gidx].unsqueeze(0)
    if causal:
        allow = allow & (k_gidx.unsqueeze(0) <= q_gidx.unsqueeze(1))
    return torch.where(allow, torch.zeros((), device=allow.device),
                       torch.full((), NEG_INF, device=allow.device))


def _repeat_kv(t, n_heads):
    # [b, s, Hkv, d] -> [b, s, Hq, d] for GQA
    if t.shape[2] == n_heads:
        return t
    rep = n_heads // t.shape[2]
    return t.repeat_interleave(rep, dim=2)


def masked_block_fwd(q, k, v, add_mask, scale):
    """(out, lse) for one block with additive doc mask. lse is -inf for query rows with
    no allowed key in this block (the online-softmax merge handles -inf)."""
    qf = q.transpose(1, 2).float()                       # [b, H, sq, d]
    kf = _repeat_kv(k, q.shape[2]).transpose(1, 2).float()
    vf = _repeat_kv(v, q.shape[2]).transpose(1, 2).float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale + add_mask  # [b, H, sq, sk]
    lse = torch.logsumexp(scores, dim=-1)                # [b, H, sq], -inf if all masked
    safe = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    p = torch.exp(scores - safe.unsqueeze(-1))           # all-masked rows -> 0
    out = torch.matmul(p, vf)                            # [b, H, sq, d]
    return out.transpose(1, 2).to(q.dtype), lse


def varlen_block_fwd(q, k, v, q_off, k_off, L, doc_ids, scale, diagonal, attn_impl, flash):
    """One packed block by global offset. flash (GPU) or explicit oracle (CPU). diagonal
    = within-range causal; off-diagonal = same-doc (caller guarantees k before q)."""
    iq = torch.arange(q_off, q_off + L, device=doc_ids.device)
    ik = torch.arange(k_off, k_off + L, device=doc_ids.device)
    if flash:
        return flash_block_fwd(q, k, v, iq, ik, doc_ids, scale, diagonal, attn_impl)
    return masked_block_fwd(q, k, v, additive_doc_mask(iq, ik, doc_ids, diagonal), scale)


def varlen_block_bwd(d_out, q, k, v, out, lse, q_off, k_off, L, doc_ids, scale,
                     diagonal, attn_impl, flash):
    iq = torch.arange(q_off, q_off + L, device=doc_ids.device)
    ik = torch.arange(k_off, k_off + L, device=doc_ids.device)
    if flash:
        return flash_block_bwd(d_out, q, k, v, out, lse, iq, ik, doc_ids, scale,
                               diagonal, attn_impl)
    return masked_block_bwd(d_out, q, k, v, out, lse,
                            additive_doc_mask(iq, ik, doc_ids, diagonal), scale)


def _straddle_index(q_gidx, k_gidx, doc_ids):
    """For an off-diagonal block (k entirely before q), the q/k positions of documents
    present in BOTH ranges, grouped by document (contiguous, ascending), plus the
    per-document cu_seqlens for each side. Returns (q_sel, k_sel, cu_q, cu_k, mq, mk)
    or None if no document straddles."""
    dq = doc_ids[q_gidx]
    dk = doc_ids[k_gidx]
    shared = dq[torch.isin(dq, dk)].unique()
    if shared.numel() == 0:
        return None
    q_sel = torch.isin(dq, shared).nonzero(as_tuple=False).view(-1)
    k_sel = torch.isin(dk, shared).nonzero(as_tuple=False).view(-1)
    _, cq = torch.unique_consecutive(dq[q_sel], return_counts=True)
    _, ck = torch.unique_consecutive(dk[k_sel], return_counts=True)
    z = torch.zeros(1, dtype=torch.int32, device=q_gidx.device)
    cu_q = torch.cat([z, cq.cumsum(0).to(torch.int32)])  # cumsum promotes to int64
    cu_k = torch.cat([z, ck.cumsum(0).to(torch.int32)])
    return q_sel, k_sel, cu_q, cu_k, int(cq.max()), int(ck.max())


def flash_block_fwd(q, k, v, q_gidx, k_gidx, doc_ids, scale, diagonal,
                    attn_implementation="flash_attention_2"):
    """Flash-varlen block. ``q/k/v``: [1, L, H, d]. Diagonal = within-range per-doc
    causal; off-diagonal = same-doc full attention over straddling docs (k entirely
    before q). Returns (out [1, L, Hq, d], lse [1, Hq, L]); lse -inf where a query has
    no allowed key (the merge handles it)."""
    fwd = _flash_varlen_ops(attn_implementation)[0]
    L, Hq, d = q.shape[1], q.shape[2], q.shape[3]
    if diagonal:
        cu, m = _seg_cu(doc_ids[q_gidx])
        out, lse, *_ = fwd(q[0], k[0], v[0], cu, cu, m, m, 0.0, scale, True)
        return out.unsqueeze(0), lse.unsqueeze(0)
    st = _straddle_index(q_gidx, k_gidx, doc_ids)
    out = q.new_zeros(L, Hq, d)
    lse = torch.full((Hq, L), NEG_INF, device=q.device, dtype=torch.float32)
    if st is not None:
        q_sel, k_sel, cu_q, cu_k, mq, mk = st
        o, l, *_ = fwd(q[0][q_sel], k[0][k_sel], v[0][k_sel], cu_q, cu_k, mq, mk,
                       0.0, scale, False)
        out[q_sel] = o
        lse[:, q_sel] = l
    return out.unsqueeze(0), lse.unsqueeze(0)


def flash_block_bwd(d_out, q, k, v, out, lse, q_gidx, k_gidx, doc_ids, scale, diagonal,
                    attn_implementation="flash_attention_2"):
    """Backward of :func:`flash_block_fwd` using the GLOBAL out/lse (ring identity).
    Returns dq, dk, dv [1, L/Hkv, ...] matching q/k/v shapes."""
    bwd = _flash_varlen_ops(attn_implementation)[1]
    dq = torch.zeros_like(q[0])
    dk = torch.zeros_like(k[0])
    dv = torch.zeros_like(v[0])
    if diagonal:
        cu, m = _seg_cu(doc_ids[q_gidx])
        bwd(d_out[0], q[0], k[0], v[0], out[0], lse[0], dq, dk, dv,
            cu, cu, m, m, 0.0, scale, True, -1, -1, 0.0, None, False)
        return dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0)
    st = _straddle_index(q_gidx, k_gidx, doc_ids)
    if st is not None:
        q_sel, k_sel, cu_q, cu_k, mq, mk = st
        dqp = torch.zeros_like(q[0][q_sel])
        dkp = torch.zeros_like(k[0][k_sel])
        dvp = torch.zeros_like(v[0][k_sel])
        bwd(d_out[0][q_sel], q[0][q_sel], k[0][k_sel], v[0][k_sel],
            out[0][q_sel], lse[0][:, q_sel], dqp, dkp, dvp,
            cu_q, cu_k, mq, mk, 0.0, scale, False, -1, -1, 0.0, None, False)
        dq[q_sel] = dqp
        dk[k_sel] = dkp
        dv[k_sel] = dvp
    return dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0)


def masked_block_bwd(d_out, q, k, v, out, lse, add_mask, scale):
    """dq, dk, dv for one block using the GLOBAL out/lse (ring identity). Each query
    attends at least its own diagonal token, so the global lse is finite (no nan)."""
    nq = q.shape[2]
    qf = q.transpose(1, 2).float()
    kf = _repeat_kv(k, nq).transpose(1, 2).float()
    vf = _repeat_kv(v, nq).transpose(1, 2).float()
    dof = d_out.transpose(1, 2).float()
    of = out.transpose(1, 2).float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale + add_mask
    p = torch.exp(scores - lse.unsqueeze(-1))            # masked -> 0
    delta = (dof * of).sum(-1, keepdim=True)
    dv = torch.matmul(p.transpose(-1, -2), dof)          # [b, H, sk, d]
    dp = torch.matmul(dof, vf.transpose(-1, -2))
    ds = p * (dp - delta)
    dq = torch.matmul(ds, kf) * scale
    dk = torch.matmul(ds.transpose(-1, -2), qf) * scale
    # fold GQA grads back to kv head count
    hkv = k.shape[2]
    if hkv != nq:
        dk = dk.unflatten(1, (hkv, nq // hkv)).sum(2)
        dv = dv.unflatten(1, (hkv, nq // hkv)).sum(2)
    return (dq.transpose(1, 2).to(q.dtype),
            dk.transpose(1, 2).to(k.dtype),
            dv.transpose(1, 2).to(v.dtype))
