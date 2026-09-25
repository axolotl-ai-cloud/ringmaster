"""Sequence sharding across the CP group (dense / v1).

Each rank gets a contiguous (or zigzag) slice with GLOBAL position_ids (RoPE is
applied per-shard before attention). Packing/varlen is v2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShardInfo:
    original_seq_len: int
    pad_len: int


def _pad_multiple(seq_len: int, cp_size: int) -> int:
    rem = seq_len % cp_size
    return 0 if rem == 0 else cp_size - rem


def cu_seqlens_from_position_ids(position_ids):
    """Global flash ``cu_seqlens`` + max_seqlen from packed ``position_ids`` (which reset
    to 0 at each document start). Returns None when not packed (<= one segment per row,
    i.e. plain dense/batched attention). Mirrors transformers'
    ``prepare_fa_kwargs_from_position_ids``."""
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=False).view(-1)
    if starts.numel() <= position_ids.shape[0]:
        return None
    cu = torch.cat(
        [
            starts.to(torch.int32),
            torch.tensor([pos.numel()], dtype=torch.int32, device=pos.device),
        ]
    )
    return cu, int(cu.diff().max().item())


def varlen_meta(global_position_ids, total_padded_len):
    """Global document boundaries, including per-row CP padding segments."""
    if global_position_ids is None:
        return None
    rows, length = global_position_ids.shape
    boundaries = []
    packed = False
    for row in range(rows):
        starts = (global_position_ids[row] == 0).nonzero().flatten()
        starts = torch.unique(torch.cat((starts.new_zeros(1), starts)))
        packed |= starts.numel() > 1
        boundaries.append(starts + row * total_padded_len)
        if total_padded_len > length:
            boundaries.append(starts.new_tensor([row * total_padded_len + length]))
    if not packed:
        return None
    cu = torch.cat(
        (*boundaries, boundaries[0].new_tensor([rows * total_padded_len]))
    ).to(torch.int32)
    return cu, int(cu.diff().max().item())


def _ensure_global_shift_labels(batch):
    """Shift once on the full sequence before sharding: per-shard the boundary target
    (first token of the next rank's shard) is unreachable and would train vs -100."""
    if "shift_labels" not in batch and batch.get("labels") is not None:
        labels = batch["labels"]
        shift = torch.full_like(labels, -100)
        shift[:, :-1] = labels[:, 1:]
        batch["shift_labels"] = shift


def _pad_to(batch, seq_len, pad_len):
    if not pad_len:
        return seq_len
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == seq_len:
            pad_value = -100 if key in ("labels", "shift_labels") else 0
            pad = torch.full(
                (val.size(0), pad_len, *val.shape[2:]),
                pad_value,
                dtype=val.dtype,
                device=val.device,
            )
            if key == "position_ids" and val.ndim == 2:
                pad = val[:, -1:] + torch.arange(
                    1, pad_len + 1, device=val.device, dtype=val.dtype
                )
            batch[key] = torch.cat([val, pad], dim=1)
    return seq_len + pad_len


def _zigzag_shard(batch, cp_rank, cp_size):
    """Zigzag (head_tail) shard: rank holds chunks [r, 2W-1-r] (balances the causal ring)."""
    bsz, seq_len = batch["input_ids"].shape[:2]
    device = batch["input_ids"].device
    pad_len = _pad_multiple(seq_len, 2 * cp_size)

    _ensure_global_shift_labels(batch)
    if batch.get("position_ids") is None:
        batch["position_ids"] = (
            torch.arange(0, seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    total = _pad_to(batch, seq_len, pad_len)
    half = total // (2 * cp_size)
    lo, hi = cp_rank, 2 * cp_size - 1 - cp_rank
    idx = torch.cat(
        [
            torch.arange(lo * half, (lo + 1) * half, device=device),
            torch.arange(hi * half, (hi + 1) * half, device=device),
        ]
    )
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == total:
            batch[key] = val.index_select(1, idx).contiguous()
    return batch, ShardInfo(original_seq_len=seq_len, pad_len=pad_len)


def shard_batch(
    batch: dict[str, torch.Tensor],
    *,
    cp_rank: int,
    cp_size: int,
    load_balance: str | None = None,
) -> tuple[dict[str, torch.Tensor], ShardInfo]:
    """Shard sequence-dim tensors in ``batch`` for this CP rank.

    ``load_balance=None`` derives the layout from the active runtime (the single
    source of truth shared with the attention, so they can't drift); falls back to
    contiguous when no runtime is active.
    """
    if cp_size == 1:
        return batch, ShardInfo(original_seq_len=batch["input_ids"].size(1), pad_len=0)
    if load_balance is None:
        from ringmaster.runtime import maybe_runtime

        rt = maybe_runtime()
        load_balance = rt.shard_load_balance if rt is not None else "contiguous"
    if load_balance == "head_tail":
        return _zigzag_shard(batch, cp_rank, cp_size)

    bsz, seq_len = batch["input_ids"].shape[:2]
    device = batch["input_ids"].device
    pad_len = _pad_multiple(seq_len, cp_size)

    _ensure_global_shift_labels(batch)
    if batch.get("position_ids") is None:
        batch["position_ids"] = (
            torch.arange(0, seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    total = _pad_to(batch, seq_len, pad_len)
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == total:
            batch[key] = val.chunk(cp_size, dim=1)[cp_rank].contiguous()

    return batch, ShardInfo(original_seq_len=seq_len, pad_len=pad_len)
