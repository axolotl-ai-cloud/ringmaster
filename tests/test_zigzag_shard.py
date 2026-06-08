"""Unit tests for zigzag (head_tail) sharding — the brittle reorder/position/shift
bookkeeping. Pure CPU, no distributed needed (we drive _zigzag_shard per rank)."""

import pytest
import torch

from ringmaster.shard import _zigzag_shard, shard_batch


def _shard_all(batch_fn, cp_size):
    """Run _zigzag_shard for every rank; return list of per-rank batches."""
    return [_zigzag_shard(batch_fn(), r, cp_size) for r in range(cp_size)]


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_len", [32, 48])  # 48 not divisible by 2*8 -> exercises pad
def test_zigzag_roundtrip_and_positions(cp_size, seq_len):
    if seq_len % (2 * cp_size) != 0:
        pad = (2 * cp_size) - (seq_len % (2 * cp_size))
    else:
        pad = 0
    total = seq_len + pad
    ids = torch.arange(seq_len).view(1, seq_len)  # token id == global position (for checking)

    shards = _shard_all(lambda: {"input_ids": ids.clone()}, cp_size)

    local_len = total // cp_size
    recon = torch.full((1, total), -1, dtype=torch.long)
    for r, (batch, info) in enumerate(shards):
        assert info.original_seq_len == seq_len and info.pad_len == pad
        local = batch["input_ids"]
        pos = batch["position_ids"]
        assert local.shape == (1, local_len), f"rank{r} {local.shape}"
        # position_ids are the GLOBAL indices this rank holds
        half = total // (2 * cp_size)
        lo, hi = r, 2 * cp_size - 1 - r
        expected_idx = list(range(lo * half, (lo + 1) * half)) + list(range(hi * half, (hi + 1) * half))
        assert pos[0].tolist() == expected_idx, f"rank{r} pos {pos[0].tolist()} != {expected_idx}"
        # scatter local tokens back to their global positions
        for p, tok in zip(pos[0].tolist(), local[0].tolist()):
            recon[0, p] = tok
    # every non-pad position reconstructed exactly once, matching the original ids
    assert torch.equal(recon[0, :seq_len], ids[0]), f"roundtrip mismatch: {recon}"


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_shift_labels(cp_size):
    seq_len = 8 * cp_size
    total = seq_len
    labels = torch.arange(100, 100 + seq_len).view(1, seq_len)
    shards = _shard_all(lambda: {"input_ids": labels.clone(), "labels": labels.clone()}, cp_size)
    # global shift target: position p predicts labels[p+1], last = -100
    global_shift = torch.full((total,), -100, dtype=torch.long)
    global_shift[:-1] = labels[0, 1:]
    for r, (batch, _) in enumerate(shards):
        assert "shift_labels" in batch
        pos = batch["position_ids"][0].tolist()
        got = batch["shift_labels"][0].tolist()
        want = [int(global_shift[p]) for p in pos]
        assert got == want, f"rank{r} shift {got} != {want}"


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_disjoint_cover(cp_size):
    """Every global position is owned by exactly one rank (disjoint + complete)."""
    seq_len = 8 * cp_size
    seen = {}
    for r in range(cp_size):
        batch, _ = _zigzag_shard({"input_ids": torch.zeros(1, seq_len, dtype=torch.long)}, r, cp_size)
        for p in batch["position_ids"][0].tolist():
            assert p not in seen, f"position {p} owned by rank {seen.get(p)} and {r}"
            seen[p] = r
    assert sorted(seen) == list(range(seq_len))


def test_shard_batch_dispatches_head_tail():
    out, _ = shard_batch(
        {"input_ids": torch.arange(16).view(1, 16), "labels": torch.arange(16).view(1, 16)},
        cp_rank=0, cp_size=2, load_balance="head_tail",
    )
    # rank0 of cp2 over 16 tokens (4 chunks of 4): holds chunks 0 and 3
    assert out["position_ids"][0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert "shift_labels" in out
