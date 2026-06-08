"""Pluggable per-block attention kernels for the ring loop.

Each returns ``(out, lse)`` with out ``[b, s, h, d]`` and lse ``[b, h, s]``:

  * ``hf_kernels_block`` — Dao-style ``flash_attn_func`` loaded from HF kernels
    (FA2/3/4) via transformers; autograd-aware, so the ring trains with no manual
    backward. No flash_attn pypi.
  * ``aten_block`` — torch's built-in flash SDPA op; forward-only (the raw aten op
    is not autograd-registered), for inference / memory-frugal paths.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def _flash_fn(attn_implementation: str):
    from transformers.modeling_flash_attention_utils import _lazy_imports

    flash_attn_func, *_ = _lazy_imports(attn_implementation)
    if flash_attn_func is None:
        raise RuntimeError(f"no flash kernel for '{attn_implementation}' from HF kernels")
    return flash_attn_func


def hf_kernels_block(q, k, v, *, causal, scaling, dropout, attn_implementation, window=None):
    """q/k/v: [b, s, h, d]. Returns (out [b,s,h,d], lse [b,h,s])."""
    flash_attn_func = _flash_fn(attn_implementation)
    kwargs = dict(
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=causal,
        return_attn_probs=True,
    )
    if window is not None:
        kwargs["window_size"] = window
    out, lse, *_ = flash_attn_func(q, k, v, **kwargs)
    return out, lse


def aten_block(q, k, v, *, causal, scaling, dropout, attn_implementation=None, window=None):
    """Forward-only block via aten flash SDPA. q/k/v: [b, s, h, d]."""
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # [b, h, s, d]
    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        qt, kt, vt, dropout_p=dropout, is_causal=causal, scale=scaling
    )
    return out.transpose(1, 2), lse  # [b, s, h, d], [b, h, s]


def math_block(q, k, v, *, causal, scaling, dropout=0.0, attn_implementation=None, window=None):
    """Reference block via explicit softmax. Autograd-aware and CPU-capable, so the
    ring/USP loops can be validated on gloo without GPUs or flash. q/k/v: [b, s, h, d]."""
    qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))  # [b, h, s, d]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(qt.shape[-1])
    scores = (qt @ kt.transpose(-1, -2)) * scale  # [b, h, sq, sk]
    sq, sk = scores.shape[-2], scores.shape[-1]
    if causal:
        # bottom-right aligned causal (handles sq <= sk blocks)
        mask = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril(sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    if window is not None:
        left, _ = window
        idx_q = torch.arange(sq, device=scores.device).unsqueeze(1) + (sk - sq)
        idx_k = torch.arange(sk, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(idx_q - idx_k > left, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [b, h, sq]
    probs = torch.exp(scores - lse.unsqueeze(-1))
    out = probs @ vt  # [b, h, sq, d]
    return out.transpose(1, 2).to(q.dtype), lse


PROVIDERS = {
    "hf_kernels": hf_kernels_block,
    "torch_native": aten_block,
    "math": math_block,
}
