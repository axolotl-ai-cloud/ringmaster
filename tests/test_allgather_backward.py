"""Flash-style detached normalizers must not lose cross-block softmax gradients."""

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("ringmaster")

from ringmaster.ring import loop  # noqa: E402
from ringmaster.ring.kernels import math_block  # noqa: E402


def test_allgather_preserves_global_softmax_gradient(monkeypatch):
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 16, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    expected, _ = math_block(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=True,
        scaling=None,
    )
    reference_grads = torch.autograd.grad(expected.sum(), (q, k, v), retain_graph=True)
    monkeypatch.setattr(dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        loop._AllGatherKV, "apply", lambda x, group: torch.stack(x.chunk(2, dim=1))
    )

    def flash_like(*args, **kwargs):
        output, normalizer = math_block(*args, **kwargs)
        return output, normalizer.detach()

    actual = loop._allgather_ring(
        q, k, v, None, True, None, 0.0, flash_like, "math", None
    )
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), (q, k, v))
    for gradient, reference in zip(gradients, reference_grads, strict=True):
        torch.testing.assert_close(gradient, reference)
