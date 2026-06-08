"""CPU unit tests for ringmaster pure logic (no distributed)."""

import pytest
import torch

from ringmaster.balancers import contiguous_indices, head_tail_indices
from ringmaster.config import AUTO, Backend, RingmasterConfig
from ringmaster.shard import shard_batch
from ringmaster.strategies.usp import auto_select


def _auto(total, kv, intra, backend=Backend.AUTO, u=AUTO, r=AUTO):
    return auto_select(
        total=total,
        requested_backend=backend,
        requested_ulysses=u,
        requested_ring=r,
        num_kv_heads=kv,
        intra_node_size=intra,
    )


def test_auto_select_pure_ulysses():
    assert _auto(8, 8, 8) == (8, 1, Backend.ULYSSES)


def test_auto_select_gqa_usp():
    assert _auto(8, 2, 8) == (2, 4, Backend.USP)


def test_auto_select_mqa_pure_ring():
    assert _auto(8, 1, 8) == (1, 8, Backend.RING)


def test_auto_select_multinode_keeps_ulysses_intra_node():
    # 2 nodes x 8, 8 kv heads -> ulysses fills a node, ring spans nodes
    assert _auto(16, 8, 8) == (8, 2, Backend.USP)


def test_auto_select_unknown_heads_defaults_full_ulysses():
    assert _auto(4, None, 4) == (4, 1, Backend.ULYSSES)


def test_explicit_ulysses_requires_head_divisibility():
    with pytest.raises(ValueError):
        _auto(8, 3, 8, backend=Backend.ULYSSES)


def test_explicit_degrees_must_multiply():
    with pytest.raises(ValueError):
        _auto(8, 8, 8, u=4, r=4)


def test_config_normalize():
    cfg = RingmasterConfig(size=8)
    cfg.normalize(num_kv_heads=2, intra_node_size=8)
    assert cfg.ulysses_size == 2 and cfg.ring_size == 4
    assert cfg.backend == Backend.USP


def test_balancer_contiguous():
    assert contiguous_indices(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_balancer_head_tail_zigzag():
    # rank r gets early block r and late block (2*world-1-r)
    assert head_tail_indices(8, 2) == [[0, 1, 6, 7], [2, 3, 4, 5]]


def test_balancer_divisibility_errors():
    with pytest.raises(ValueError):
        contiguous_indices(7, 2)
    with pytest.raises(ValueError):
        head_tail_indices(8, 3)


def test_shard_batch_noop_when_disabled():
    batch = {"input_ids": torch.arange(8).view(1, 8)}
    out, info = shard_batch(batch, cp_rank=0, cp_size=1)
    assert info.pad_len == 0 and out["input_ids"].shape == (1, 8)


def test_shard_batch_contiguous_split_and_global_positions():
    base = {
        "input_ids": torch.arange(8).view(1, 8),
        "labels": torch.arange(8).view(1, 8),
    }
    shard0, _ = shard_batch({k: v.clone() for k, v in base.items()}, cp_rank=0, cp_size=2)
    shard1, _ = shard_batch({k: v.clone() for k, v in base.items()}, cp_rank=1, cp_size=2)

    assert torch.equal(shard0["input_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(shard1["input_ids"], torch.tensor([[4, 5, 6, 7]]))
    # position ids are global per shard (RoPE correctness)
    assert torch.equal(shard0["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(shard1["position_ids"], torch.tensor([[4, 5, 6, 7]]))


def test_recurrent_mixer_predicate():
    from ringmaster.strategies.state_passing import is_recurrent_mixer

    class Mamba2Mixer:
        pass

    class Qwen3NextGatedDeltaNet:
        pass

    class LlamaAttention:
        pass

    assert is_recurrent_mixer(Mamba2Mixer())
    assert is_recurrent_mixer(Qwen3NextGatedDeltaNet())
    assert not is_recurrent_mixer(LlamaAttention())


def test_shard_batch_pads_to_multiple():
    base = {"input_ids": torch.arange(6).view(1, 6), "labels": torch.arange(6).view(1, 6)}
    shard0, info = shard_batch(
        {k: v.clone() for k, v in base.items()}, cp_rank=0, cp_size=4
    )
    # 6 -> pad to 8, each of 4 ranks gets 2
    assert info.original_seq_len == 6 and info.pad_len == 2
    assert shard0["input_ids"].shape == (1, 2)
