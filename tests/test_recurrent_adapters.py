"""Preflight rejection and instance-local kernel binding contracts."""

import pytest
import torch
from ringmaster import recurrent


def chunk_kda(**kwargs):
    return kwargs["q"]


class KimiDeltaAttention(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
        return chunk_kda(
            q=hidden_states,
            k=hidden_states,
            v=hidden_states,
            g=hidden_states,
            beta=hidden_states,
            **kwargs,
        )


def test_kda_detection_is_structural():
    mixer = KimiDeltaAttention()
    assert recurrent.kda_mixers(torch.nn.Sequential(mixer)) == [mixer]


def test_kda_wiring_guards_cache_and_restores(monkeypatch):
    from types import SimpleNamespace

    mixer = KimiDeltaAttention()
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(mixer, name, torch.nn.Identity())
    monkeypatch.setattr(recurrent, "validate_kda", lambda mixers: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    monkeypatch.setattr(
        recurrent,
        "require_kda_cp",
        lambda: (
            lambda cu, **kwargs: SimpleNamespace(cu=cu),
            lambda q, *args, **kwargs: (q * 2, None),
            lambda *args, **kwargs: None,
        ),
    )
    original = mixer.forward.__func__
    restore = recurrent.wire_kda([mixer], object())
    x = torch.ones(1, 8, 4)
    assert torch.equal(mixer(x)[0], x * 2)
    with pytest.raises(ValueError, match="use_cache=False"):
        mixer(x, None, object())
    with pytest.raises(ValueError, match="global attention mask"):
        mixer(x, attention_mask=torch.zeros(1, 8))
    with pytest.raises(ValueError, match="batch size"):
        mixer(x.expand(2, -1, -1))
    restore()
    assert mixer.forward.__func__ is original
    assert "forward" not in vars(mixer.q_conv1d)
    assert torch.equal(mixer(x), x)


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[0, 1, 1, 1]]),
        torch.tensor([[1, 0, 1, 0]]),
        torch.ones(1, 1, 4, 4),
        torch.tensor([[1.0, 0.5, 1.0, 1.0]]),
        torch.tensor([[1.0, float("nan"), 1.0, 1.0]]),
    ],
)
def test_cp_rejects_unsupported_global_masks(monkeypatch, mask):
    from ringmaster import sp_context

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(sp_context, "broadcast_batch", lambda batch, group: None)
    manager = sp_context.ContextParallelContextManager([], object())
    hook = manager._make_pre_hook(["input_ids", "attention_mask"])
    with pytest.raises(ValueError, match="right-padded causal"):
        hook(
            torch.nn.Identity(),
            (),
            {"input_ids": torch.ones(1, 4), "attention_mask": mask},
        )


def test_recurrent_plan_includes_owned_convolution_children(monkeypatch):
    from ringmaster.strategies.state_passing import recurrent_plan

    class ShortConvolution(torch.nn.Identity):
        pass

    mixer = KimiDeltaAttention()
    mixer.q_conv1d = ShortConvolution()
    monkeypatch.setattr(recurrent, "validate_kda", lambda mixers: None)
    assert recurrent_plan([torch.nn.Sequential(mixer)], 4) == ([], [mixer], [])


def test_disabled_cp_wiring_needs_no_process_group():
    import ringmaster as rm

    rm.setup(rm.RingmasterConfig(size=1))
    try:
        mixer = KimiDeltaAttention()
        original = mixer.forward.__func__
        wiring = rm.wire_recurrent_layers(mixer)
        assert not wiring
        wiring.restore()
        assert mixer.forward.__func__ is original
    finally:
        rm.teardown()


def test_fla_global_boundaries_override_local_cu():
    from types import SimpleNamespace

    from ringmaster.runtime import maybe_runtime, set_runtime

    previous = maybe_runtime()
    x = torch.ones(1, 8, 4)
    global_cu = torch.tensor([0, 5, 19, 20, 32], dtype=torch.int32)
    try:
        set_runtime(SimpleNamespace(varlen=(global_cu, 14)))
        actual = recurrent._global_cu_seqlens(x, 4, torch.tensor([0, 5, 8]))
        assert actual.tolist() == global_cu.tolist()
        assert actual.dtype == torch.long
        set_runtime(None)
        with pytest.raises(ValueError, match="global document boundaries"):
            recurrent._global_cu_seqlens(x, 4, torch.tensor([0, 5, 8]))
    finally:
        set_runtime(previous)
