"""Validate the flash-varlen block kernels (FA2/FA3/FA4 from hf `kernels`) against the
explicit document-masked oracle — forward + backward, diagonal + off-diagonal, GQA.

Single GPU, no distributed. Each FA kernel that isn't installed/runnable is skipped.
The explicit fp32 block (masked_block_fwd/bwd) is the ground truth; the flash blocks
must match it (the kernels are what `varlen_block_*` dispatches to on GPU).
"""

import pytest
import torch

from ringmaster.ring.varlen_blocks import (
    _flash_varlen_ops,
    additive_doc_mask,
    flash_block_bwd,
    flash_block_fwd,
    masked_block_bwd,
    masked_block_fwd,
)

CUDA = torch.cuda.is_available()
IMPLS = ["flash_attention_2", "flash_attention_3", "flash_attention_4"]


def _available(impl):
    if not CUDA:
        return False
    try:
        return _flash_varlen_ops(impl) is not None
    except Exception:
        return False


def _rand(L, H, d, dev):
    return torch.randn(1, L, H, d, device=dev, dtype=torch.bfloat16, requires_grad=False)


def _close(a, b, atol=2e-2, rtol=4e-2):
    """allclose-style: tolerant of bf16 ULPs at large gradient magnitudes while still
    catching constant offsets / wrong math."""
    af, bf = a.float(), b.float()
    return bool(((af - bf).abs() <= atol + rtol * bf.abs()).all())


@pytest.mark.skipif(not CUDA, reason="flash varlen kernels need CUDA")
@pytest.mark.parametrize("impl", IMPLS)
def test_flash_block_matches_explicit(impl):
    if not _available(impl):
        pytest.skip(f"{impl} not available")
    dev = "cuda"
    L, Hq, Hkv, d = 64, 8, 2, 64
    scale = 1.0 / (d ** 0.5)
    torch.manual_seed(0)

    # ---- diagonal: range [0,L), multiple documents ----
    doc = torch.tensor([0] * 20 + [1] * 20 + [2] * 24, device=dev)
    gidx = torch.arange(L, device=dev)
    q, k, v = _rand(L, Hq, d, dev), _rand(L, Hkv, d, dev), _rand(L, Hkv, d, dev)
    add = additive_doc_mask(gidx, gidx, doc, causal=True)
    o_e, l_e = masked_block_fwd(q, k, v, add, scale)
    o_f, l_f = flash_block_fwd(q, k, v, gidx, gidx, doc, scale, True, impl)
    assert _close(o_f, o_e), f"{impl} diag fwd out"
    assert _close(l_f, l_e), f"{impl} diag fwd lse"

    d_out = _rand(L, Hq, d, dev)
    dqe, dke, dve = masked_block_bwd(d_out, q, k, v, o_e, l_e, add, scale)
    dqf, dkf, dvf = flash_block_bwd(d_out, q, k, v, o_e.to(q.dtype), l_e,
                                    gidx, gidx, doc, scale, True, impl)
    assert _close(dqf, dqe), f"{impl} diag dq"
    assert _close(dkf, dke), f"{impl} diag dk"
    assert _close(dvf, dve), f"{impl} diag dv"

    # ---- off-diagonal: q=[L,2L) k=[0,L), one document spans both (all straddle) ----
    doc2 = torch.zeros(2 * L, device=dev, dtype=torch.long)  # single doc over [0,2L)
    q_gidx = torch.arange(L, 2 * L, device=dev)
    k_gidx = torch.arange(0, L, device=dev)
    q2, k2, v2 = _rand(L, Hq, d, dev), _rand(L, Hkv, d, dev), _rand(L, Hkv, d, dev)
    add2 = additive_doc_mask(q_gidx, k_gidx, doc2, causal=False)  # k all before q
    oe2, le2 = masked_block_fwd(q2, k2, v2, add2, scale)
    of2, lf2 = flash_block_fwd(q2, k2, v2, q_gidx, k_gidx, doc2, scale, False, impl)
    assert _close(of2, oe2), f"{impl} offdiag fwd out"
    assert _close(lf2, le2), f"{impl} offdiag fwd lse"

    do2 = _rand(L, Hq, d, dev)
    dqe2, dke2, dve2 = masked_block_bwd(do2, q2, k2, v2, oe2, le2, add2, scale)
    dqf2, dkf2, dvf2 = flash_block_bwd(do2, q2, k2, v2, oe2.to(q2.dtype), le2,
                                       q_gidx, k_gidx, doc2, scale, False, impl)
    assert _close(dqf2, dqe2), f"{impl} offdiag dq"
    assert _close(dkf2, dke2), f"{impl} offdiag dk"
    assert _close(dvf2, dve2), f"{impl} offdiag dv"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_at_least_fa2_available():
    assert _available("flash_attention_2"), "flash_attention_2 varlen ops must load"
