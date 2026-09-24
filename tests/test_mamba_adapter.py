"""Mamba's fused normalization must survive decomposition into CP kernels."""

from types import FunctionType, MethodType, SimpleNamespace

import pytest
import torch
from ringmaster.mamba import _norm_forward, _uses_fused_norm


def _fused():
    pass


_fused.__name__ = "mamba_split_conv1d_scan_combined"
_fused.__module__ = "mamba_ssm.ops.triton.ssd_combined"


def _fallback():
    return None


_fallback.__name__ = "mamba2_split_conv1d_scan_combined"
_fallback.__module__ = "transformers.models.mamba2.modeling_mamba2"


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.variance_epsilon = 1e-5

    def forward(self, x, gate=None):
        if gate is not None:
            x = x * torch.nn.functional.silu(gate)
        return (
            self.weight
            * x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        )


@pytest.mark.parametrize("training,fused", [(True, True), (True, False), (False, True)])
def test_fused_grouped_norm_preserves_training_and_eval(training, fused):
    norm = Norm()
    mixer = SimpleNamespace(norm=norm, n_groups=2, training=training)
    adapted = _norm_forward(mixer, _fused if fused else _fallback)
    x = torch.tensor([[[1.0, 2.0, 10.0, 20.0]]], requires_grad=True)
    gate = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]], requires_grad=True)
    actual = adapted(x, gate)
    if training and fused:
        gated = x * torch.nn.functional.silu(gate)
        groups = gated.chunk(2, dim=-1)
        expected = (
            torch.cat(
                [
                    group
                    * torch.rsqrt(
                        group.square().mean(-1, keepdim=True) + norm.variance_epsilon
                    )
                    for group in groups
                ],
                dim=-1,
            )
            * norm.weight
        )
    else:
        expected = norm(x, gate)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(
        actual.sum(), (x, gate, norm.weight), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.sum(), (x, gate, norm.weight))
    for actual_item, expected_item in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_item, expected_item)


def test_native_package_and_hub_dispatch_detection():
    def wrapper(implementation):
        def call(*args, **kwargs):
            return implementation(*args, **kwargs)

        return call

    assert _uses_fused_norm(wrapper(_fused))
    assert not _uses_fused_norm(wrapper(_fallback))
    selected = SimpleNamespace(forward=MethodType(wrapper(_fused), object()))
    assert _uses_fused_norm(selected)
    with pytest.raises(ValueError, match="Cannot determine"):
        _uses_fused_norm(lambda *args: None)


def test_wiring_restores_norm_and_mixer(monkeypatch):
    import ringmaster.mamba as adapter

    class Mixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Norm()
            self.n_groups = 2

        def forward(self, x):
            return self.norm(x)

    mixer = Mixer()
    original = mixer.forward.__func__
    rebound = FunctionType(
        original.__code__,
        original.__globals__
        | {
            "mamba2_split_conv1d_scan_combined": _fused,
        },
    )
    mixer.forward = MethodType(rebound, mixer)
    monkeypatch.setattr(adapter, "_bindings", lambda *args: {})
    restore = adapter.wire_mamba2([mixer], None)
    assert "forward" in vars(mixer.norm)
    restore()
    assert "forward" not in vars(mixer.norm)
    assert mixer.forward.__func__ is rebound

    def fail(*args):
        raise RuntimeError("binding failure")

    monkeypatch.setattr(adapter, "_bindings", fail)
    with pytest.raises(RuntimeError, match="binding failure"):
        adapter.wire_mamba2([mixer], None)
    assert "forward" not in vars(mixer.norm)
    assert mixer.forward.__func__ is rebound
