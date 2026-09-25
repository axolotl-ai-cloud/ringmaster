"""Ulysses sequence parallelism by *wrapping* an existing kernel.

Registers an ``AttentionInterface`` fn that all-to-alls sequence-sharded q/k/v into
head-sharded full-sequence tensors, calls the model's own kernel (FA2/3/4/sdpa/flex
from ``ALL_ATTENTION_FUNCTIONS``), then all-to-alls back — no custom CUDA, no
flash_attn pypi. Each rank computes the full causal triangle for a head subset, so
Ulysses is inherently balanced (zigzag is a Ring-only concern).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ringmaster.comm import seq_all_to_all
from ringmaster.runtime import get_runtime

REGISTERED_NAME = "ringmaster_ulysses"


def _resolve_inner(inner_name: str):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    fn = ALL_ATTENTION_FUNCTIONS.get(inner_name)
    if fn is None:
        raise ValueError(
            f"ringmaster Ulysses inner attention '{inner_name}' not found in "
            f"ALL_ATTENTION_FUNCTIONS; expected one of flash_attention_2/3/4, sdpa, "
            f"flex_attention."
        )
    return fn


@lru_cache(maxsize=2)
def _flash_varlen_fn(inner_name: str):
    from transformers.modeling_flash_attention_utils import _lazy_imports

    fn = _lazy_imports(inner_name)[1]
    if fn is None:
        raise RuntimeError(
            f"varlen CP needs a flash kernel; '{inner_name}' has no flash_attn_varlen_func"
        )
    return fn


def _ulysses_varlen(
    q, k, v, varlen, dropout, scaling, is_causal, inner_name, sliding_window=None
):
    """Packed-sequence attention on the head-sharded FULL sequence. After the Ulysses
    all-to-all each rank holds the whole pack for a head subset, so flash varlen with
    the GLOBAL cu_seqlens is exact. q/k/v: [B, Hsub, S, d]; returns [B, S, Hq, d]."""
    cu, max_len = varlen
    cu = cu.to(q.device)

    if inner_name in ("sdpa", "flex_attention"):
        lengths = cu.diff().long()
        doc = torch.repeat_interleave(
            torch.arange(len(lengths), device=q.device), lengths
        ).reshape(q.shape[0], q.shape[2])
        pos = torch.arange(q.shape[2], device=q.device)
        allowed = doc[:, :, None] == doc[:, None, :]
        if is_causal is not False:
            allowed &= pos[:, None] >= pos[None, :]
        if sliding_window:
            allowed &= pos[:, None] - pos[None, :] < sliding_window
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[:, None],
            dropout_p=dropout,
            scale=scaling,
            enable_gqa=q.shape[1] != k.shape[1],
        ).transpose(1, 2)

    def flat(t):
        return t.transpose(1, 2).reshape(-1, t.shape[1], t.shape[-1]).contiguous()

    out = _flash_varlen_fn(inner_name)(
        flat(q),
        flat(k),
        flat(v),
        cu,
        cu,
        max_len,
        max_len,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True if is_causal is None else is_causal,
        **({"window_size": (sliding_window - 1, 0)} if sliding_window else {}),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3])


def make_ulysses_attention(inner_name: str):
    """Build the AttentionInterface-compatible Ulysses forward for ``inner_name``."""

    def ulysses_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs,
    ):
        inner = _resolve_inner(inner_name)
        rt = get_runtime()
        group = rt.ulysses_group
        import torch.distributed as dist

        world = dist.get_world_size(group) if group is not None else 1
        if world == 1:
            return inner(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                **kwargs,
            )

        if attention_mask is not None:
            raise NotImplementedError(
                "ringmaster Ulysses v1 supports dense causal attention only "
                "(attention_mask must be None / is_causal=True). Padded/packed masks "
                "are v2."
            )

        # q/k/v: [b, n_heads, s_local, d]. Heads must divide the Ulysses degree;
        # the auto-selector guarantees this by choosing ulysses_size | num_kv_heads.
        for name, t in (("query", query), ("key", key), ("value", value)):
            if t.size(1) % world != 0:
                raise ValueError(
                    f"ringmaster Ulysses: {name} head count {t.size(1)} not divisible "
                    f"by ulysses_size {world}; lower ulysses_size or use backend=ring/usp."
                )

        # [b, H, s/P, d] -> [b, H/P, s, d] (scatter heads, gather sequence). Fuse the
        # collectives: MHA moves q/k/v in one all-to-all; GQA fuses just K/V.
        if query.shape[1] == key.shape[1]:
            qkv = seq_all_to_all(
                torch.stack((query, key, value), dim=0),
                scatter_dim=2,
                gather_dim=3,
                group=group,
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = seq_all_to_all(query, scatter_dim=1, gather_dim=2, group=group)
            kv = seq_all_to_all(
                torch.stack((key, value), dim=0),
                scatter_dim=2,
                gather_dim=3,
                group=group,
            )
            k, v = kv[0], kv[1]

        if rt.varlen is not None:
            # packed sequences: flash varlen over the gathered full pack (global cu_seqlens)
            attn_out = _ulysses_varlen(
                q,
                k,
                v,
                rt.varlen,
                dropout,
                scaling,
                is_causal,
                inner_name,
                kwargs.get("sliding_window"),
            )
        else:
            # The flash integration reads module.config._attn_implementation to pick the
            # kernel; it's currently our registered name, so restore the real inner kernel
            # name for the duration of the inner call (dispatch already resolved to us).
            cfg = getattr(module, "config", None)
            saved = (
                getattr(cfg, "_attn_implementation", None) if cfg is not None else None
            )
            if cfg is not None:
                cfg._attn_implementation = inner_name
            try:
                attn_out, _ = inner(
                    module,
                    q,
                    k,
                    v,
                    None,
                    dropout=dropout,
                    scaling=scaling,
                    is_causal=True if is_causal is None else is_causal,
                    **kwargs,
                )
            finally:
                if cfg is not None:
                    cfg._attn_implementation = saved

        # inner returns [b, s, H/P, d]; scatter sequence, gather heads -> [b, s/P, H, d]
        attn_out = seq_all_to_all(
            attn_out.contiguous(), scatter_dim=1, gather_dim=2, group=group
        )
        return attn_out, None

    return ulysses_attention_forward


def register_ulysses(inner_name: str, *, name: str = REGISTERED_NAME) -> str:
    """Register the Ulysses wrapper under ``name`` (set as the model's
    attn_implementation) and return it."""
    from transformers import AttentionInterface

    AttentionInterface.register(name, make_ulysses_attention(inner_name))
    return name
