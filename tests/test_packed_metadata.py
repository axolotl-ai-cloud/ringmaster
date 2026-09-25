"""Document boundaries survive CP padding independently in every batch row."""

import pytest
import torch

from ringmaster.shard import shard_batch, varlen_meta
from ringmaster.strategies.ulysses import _ulysses_varlen


def test_padded_multirow_boundaries():
    positions = torch.tensor([[0, 1, 0, 1, 2], [0, 0, 1, 2, 3]])
    cu, maximum = varlen_meta(positions, 8)
    assert cu.tolist() == [0, 2, 5, 8, 9, 13, 16]
    assert maximum == 4


def test_packed_shift_labels_preserve_rank_boundary():
    batch = {
        "input_ids": torch.arange(7)[None],
        "labels": torch.tensor([[-100, 1, 2, -100, 4, 5, 6]]),
    }
    shard, info = shard_batch(batch, cp_rank=0, cp_size=2, load_balance="none")
    assert shard["shift_labels"].tolist() == [[1, 2, -100, 4]]
    assert info.pad_len == 1


def test_multirow_ulysses_sdpa_matches_independent_documents():
    torch.manual_seed(123)
    positions = torch.tensor([[0, 1, 0, 1, 2], [0, 0, 1, 2, 3]])
    q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    out = _ulysses_varlen(q, k, v, varlen_meta(positions, 5), 0.0, None, True, "sdpa")
    references = []
    for row, lengths in enumerate(([2, 3], [1, 4])):
        parts, start = [], 0
        for length in lengths:
            parts.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[row : row + 1, :, start : start + length],
                    k[row : row + 1, :, start : start + length],
                    v[row : row + 1, :, start : start + length],
                    is_causal=True,
                    enable_gqa=True,
                )
            )
            start += length
        references.append(torch.cat(parts, dim=2))
    expected = torch.cat(references).transpose(1, 2)
    torch.testing.assert_close(out, expected)
    grad = torch.randn_like(out)
    actual_grad = torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
    reference_grad = torch.autograd.grad(expected, (q, k, v), grad)
    for actual, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("positions", [False, True])
def test_context_consumes_global_collator_boundaries(monkeypatch, positions):
    from types import SimpleNamespace

    from ringmaster import sp_context
    from ringmaster.runtime import maybe_runtime, set_runtime

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(sp_context, "broadcast_batch", lambda batch, group: None)
    manager = sp_context.ContextParallelContextManager([], object())
    hook = manager._make_pre_hook(["input_ids"])
    previous = maybe_runtime()
    runtime = SimpleNamespace(varlen=None)
    try:
        set_runtime(runtime)
        from transformers import DataCollatorWithFlattening

        collator = DataCollatorWithFlattening(
            return_position_ids=positions,
            return_flash_attn_kwargs=True,
        )
        batch = collator([{"input_ids": [0, 1, 2]}, {"input_ids": [3, 4, 5, 6]}])
        monkeypatch.setattr(
            sp_context, "broadcast_batch", lambda values, group: values.update(batch)
        )
        _, local = hook(
            torch.nn.Identity(), (), {"input_ids": torch.zeros(1, 2, dtype=torch.long)}
        )
        assert local["position_ids"].tolist() == [[0, 1, 2, 0]]
        assert "cu_seq_lens_q" not in local
        assert "max_length_q" not in local
        assert runtime.varlen[0].tolist() == [0, 3, 7, 8]
    finally:
        set_runtime(previous)
