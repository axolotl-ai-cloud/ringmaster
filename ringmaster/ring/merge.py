"""Online-softmax merge for combining per-KV-block attention outputs.

Each KV block produces a partial output and its log-sum-exp (LSE). Merging two
partials is the flash-attention rescale: weight each by ``exp(lse - new_lse)``
where ``new_lse = logaddexp(lse_a, lse_b)``. Accumulate in fp32 for stability.
"""

from __future__ import annotations

import torch


def _lse_to_bshd(block_lse: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Flash returns LSE as [b, h, s]; reshape to [b, s, h, 1] to match [b, s, h, d]."""
    if block_lse.dim() == 3 and block_lse.shape[1] == like.shape[2]:
        block_lse = block_lse.transpose(1, 2)  # [b, s, h]
    return block_lse.unsqueeze(-1).contiguous().to(torch.float32)


def update_out_and_lse(out, lse, block_out: torch.Tensor, block_lse: torch.Tensor):
    """Merge one block's (out, lse) into the running (out, lse).

    out/block_out: [b, s, h, d]; lse: [b, s, h, 1]; block_lse: [b, h, s].
    """
    block_lse = _lse_to_bshd(block_lse, block_out)
    block_out = block_out.to(torch.float32)
    if out is None:
        return block_out, block_lse
    new_lse = torch.logaddexp(lse, block_lse)
    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    return out, new_lse
