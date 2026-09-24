"""Ring backend: register a kernel-agnostic ring attention (Phase 2).

The ring loop (rotation + online-softmax LSE merge) lives in ``ringmaster.ring``;
the per-block kernel is pluggable so Ring is not limited to torch SDPA/flex:

  * ``hf_kernels`` — Dao-style flash from HF kernels (FA2/3/4), autograd-aware
    (trains), no flash_attn pypi.
  * ``torch_native`` — aten flash SDPA op (forward-only, inference/memory paths).
  * ``ring_flash_attn`` — legacy flash_attn pypi bridge (opt-in, not wired here).
"""

from __future__ import annotations

import torch

from ringmaster.config import RingImpl, RotateMethod

REGISTERED_NAME = "ringmaster_ring"


def resolve_ring_impl(ring_impl: RingImpl, inner_attn: str) -> RingImpl:
    if ring_impl != RingImpl.AUTO:
        return ring_impl
    if inner_attn.startswith("flash_attention"):
        return RingImpl.HF_KERNELS
    return RingImpl.TORCH_NATIVE


_PROVIDER = {
    RingImpl.HF_KERNELS: "hf_kernels",
    RingImpl.TORCH_NATIVE: "torch_native",
}


def make_ring_attention(provider: str, attn_implementation: str, rotate_method: RotateMethod):
    from ringmaster.ring import ring_attention
    from ringmaster.runtime import get_runtime

    def ring_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        sliding_window: int | None = None,
        **kwargs,
    ):
        rt = get_runtime()
        group = rt.ring_group
        causal = True if is_causal is None else is_causal
        window = (sliding_window - 1, 0) if sliding_window else None
        import torch.distributed as dist

        from ringmaster.config import LoadBalance

        multi = group is not None and dist.get_world_size(group) > 1
        if multi and attention_mask is not None:
            raise ValueError("Ring attention requires an unpadded causal sequence; attention_mask is unsupported")
        if multi and rt.config.load_balance in (LoadBalance.HEAD_TAIL, LoadBalance.DISTFLASH):
            if dropout or window is not None or attn_implementation not in ("flash_attention_2", "math"):
                raise ValueError("Balanced Ring requires FA2, zero dropout, and no sliding window")
        # Packed sequences: distflash keeps its balanced schedule with doc-masked
        # blocks; plain ring (and zigzag, for now) use the contiguous doc-masked path.
        if rt.varlen is not None and multi:
            cu = rt.varlen[0]
            if rt.config.load_balance == LoadBalance.DISTFLASH:
                from ringmaster.ring.distflash import distflash_attention

                return distflash_attention(
                    query, key, value, group=group, scaling=scaling, cu_seqlens=cu,
                    attn_implementation=attn_implementation,
                ), None
            if rt.config.load_balance == LoadBalance.HEAD_TAIL:
                from ringmaster.ring.zigzag import zigzag_ring_attention

                return zigzag_ring_attention(
                    query, key, value, group=group, scaling=scaling, cu_seqlens=cu
                ), None
            from ringmaster.ring.loop import varlen_ring_attention

            return varlen_ring_attention(
                query, key, value, group=group, scaling=scaling, cu_seqlens=cu
            ), None
        balanced = causal and window is None and multi
        # Zigzag (head_tail): inputs are zigzag-sharded (rank holds chunks [r, 2W-1-r]);
        # balanced half-work ring. NOT SSM-safe (permutes tokens).
        if balanced and rt.config.load_balance == LoadBalance.HEAD_TAIL:
            from ringmaster.ring.zigzag import zigzag_ring_attention

            return zigzag_ring_attention(query, key, value, group=group, scaling=scaling), None
        # DistFlashAttn-style: contiguous (SSM-safe) + balanced by routing work to idle
        # ranks (rotates KV+Q+partial-O).
        if balanced and rt.config.load_balance == LoadBalance.DISTFLASH:
            from ringmaster.ring.distflash import distflash_attention

            return distflash_attention(query, key, value, group=group, scaling=scaling), None
        out = ring_attention(
            query,
            key,
            value,
            group=group,
            causal=causal,
            scaling=scaling,
            dropout=dropout,
            provider=provider,
            rotate_method=rotate_method,
            attn_implementation=attn_implementation,
            window=window,
        )
        return out, None

    return ring_attention_forward


def register_ring(
    ring_impl: RingImpl,
    inner_attn: str,
    rotate_method: RotateMethod,
    *,
    name: str = REGISTERED_NAME,
) -> str:
    from transformers import AttentionInterface

    resolved = resolve_ring_impl(ring_impl, inner_attn)
    if resolved == RingImpl.RING_FLASH_ATTN:
        raise NotImplementedError(
            "ring_impl=ring_flash_attn (legacy flash_attn pypi) is not wired; use "
            "auto/hf_kernels/torch_native."
        )
    provider = _PROVIDER[resolved]
    # hf_kernels uses the flash kernel for blocks; torch_native uses aten flash.
    attn_for_blocks = inner_attn if resolved == RingImpl.HF_KERNELS else "flash_attention_2"
    AttentionInterface.register(
        name, make_ring_attention(provider, attn_for_blocks, rotate_method)
    )
    return name
