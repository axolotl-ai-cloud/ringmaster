"""USP (Unified Sequence Parallelism) composition + the auto-selector.

Factors the CP degree into ``ulysses_size x ring_size``: Ulysses (head all-to-all,
cheapest comm but capped by KV heads, best intra-node) composed with Ring (scales
without a head cap, heavier constant-in-P comm, best inter-node). The auto-selector
picks the split from (KV heads, total degree, intra-node size).
"""

from __future__ import annotations

from ringmaster.config import AUTO, Backend, RingImpl


def _largest_divisor(total: int, *, max_value: int, must_divide: int | None) -> int:
    """Largest d | total with d <= max_value and (must_divide % d == 0 if given)."""
    best = 1
    for d in range(1, total + 1):
        if total % d != 0:
            continue
        if d > max_value:
            continue
        if must_divide is not None and must_divide % d != 0:
            continue
        best = d
    return best


def auto_select(
    *,
    total: int,
    requested_backend: Backend,
    requested_ulysses: int,
    requested_ring: int,
    num_kv_heads: int | None,
    intra_node_size: int | None,
) -> tuple[int, int, Backend]:
    """Return ``(ulysses_size, ring_size, resolved_backend)``.

    Pure function (no distributed calls) so it can be unit-tested directly.
    """
    if total <= 1:
        return 1, 1, Backend.ULYSSES

    # Explicit degrees win and just need to be consistent.
    if requested_ulysses != AUTO or requested_ring != AUTO:
        u = requested_ulysses if requested_ulysses != AUTO else total // requested_ring
        r = requested_ring if requested_ring != AUTO else total // requested_ulysses
        if u * r != total:
            raise ValueError(
                f"ulysses_size({u}) * ring_size({r}) != context_parallel size({total})"
            )
        return u, r, _backend_for(u, r)

    cap = intra_node_size if intra_node_size else total

    if requested_backend == Backend.ULYSSES:
        if num_kv_heads is not None and num_kv_heads % total != 0:
            raise ValueError(
                f"backend=ulysses needs context_parallel size ({total}) to divide "
                f"num_kv_heads ({num_kv_heads}); use backend=usp/ring or lower the size."
            )
        return total, 1, Backend.ULYSSES

    if requested_backend == Backend.RING:
        return 1, total, Backend.RING

    # AUTO (and USP): prefer the largest Ulysses leg that fits the head budget and
    # stays intra-node, then let Ring cover the remainder.
    u = _largest_divisor(total, max_value=cap, must_divide=num_kv_heads)
    r = total // u
    return u, r, _backend_for(u, r)


def _backend_for(ulysses: int, ring: int) -> Backend:
    if ring == 1:
        return Backend.ULYSSES
    if ulysses == 1:
        return Backend.RING
    return Backend.USP


REGISTERED_NAME = "ringmaster_usp"


def make_usp_attention(provider: str, attn_implementation: str, rotate_method):
    """USP forward: Ulysses all-to-all (heads) wrapping ring attention (sequence).

    Each rank holds ``[b, H, s_local, d]`` with ``s_local = S/(U*R)``. The Ulysses
    all-to-all over the ulysses group turns this into ``[b, H/U, S/R, d]``; ring
    attention over the ring group attends across the S/R shards; a second all-to-all
    restores ``[b, s_local, H, d]``.
    """
    import torch

    from ringmaster.comm import seq_all_to_all
    from ringmaster.ring import ring_attention
    from ringmaster.runtime import get_runtime

    def usp_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout: float = 0.0,
        scaling=None,
        is_causal=None,
        sliding_window=None,
        **kwargs,
    ):
        rt = get_runtime()
        ug, rg = rt.ulysses_group, rt.ring_group
        causal = True if is_causal is None else is_causal
        window = (sliding_window - 1, 0) if sliding_window else None

        q = seq_all_to_all(query, scatter_dim=1, gather_dim=2, group=ug)
        kv = torch.stack((key, value), dim=0)
        kv = seq_all_to_all(kv, scatter_dim=2, gather_dim=3, group=ug)
        k, v = kv[0], kv[1]

        if rt.varlen is not None:
            from ringmaster.ring.loop import varlen_ring_attention

            out = varlen_ring_attention(
                q,
                k,
                v,
                group=rg,
                scaling=scaling,
                cu_seqlens=rt.varlen[0],
                causal=causal,
                dropout=dropout,
                window=window,
                attn_implementation=attn_implementation,
            )
        else:
            out = ring_attention(
                q,
                k,
                v,
                group=rg,
                causal=causal,
                scaling=scaling,
                dropout=dropout,
                provider=provider,
                rotate_method=rotate_method,
                attn_implementation=attn_implementation,
                window=window,
            )  # [b, S/R, H/U, d]
        out = seq_all_to_all(out, scatter_dim=1, gather_dim=2, group=ug)
        return out, None

    return usp_attention_forward


def register_usp(config, inner_attn: str, *, name: str = REGISTERED_NAME) -> str:
    from transformers import AttentionInterface

    from ringmaster.strategies.ring import _PROVIDER, resolve_ring_impl

    resolved = resolve_ring_impl(config.ring_impl, inner_attn)
    provider = _PROVIDER[resolved]
    attn_for_blocks = (
        inner_attn if resolved == RingImpl.HF_KERNELS else "flash_attention_2"
    )
    AttentionInterface.register(
        name, make_usp_attention(provider, attn_for_blocks, config.rotate_method)
    )
    return name
