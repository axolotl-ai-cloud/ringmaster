"""CPU/gloo correctness for packed-sequence (varlen) CP.

`varlen_ring_attention` is the unified packed path that ring / USP / zigzag /
distflash all route to under packing (contiguous shard + document-masked attention).
We validate it against an INDEPENDENT per-document causal reference (each document
attended on its own with standard causal SDPA) across several packing patterns and
world sizes — gloo, no GPU needed (the Ulysses flash-varlen path is GPU-only and
covered in test_varlen_gpu.py).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from ringmaster.ring.loop import varlen_ring_attention
from ringmaster.shard import cu_seqlens_from_position_ids, varlen_meta

# (seq_len, doc_lengths, world) — seq_len divisible by world; docs sum to seq_len.
CASES = [
    (16, [6, 10], 2),
    (16, [4, 4, 4, 4], 2),
    (24, [5, 7, 12], 2),
    (16, [16], 2),          # single doc == dense causal
    (32, [10, 9, 13], 4),
    (24, [3, 3, 6, 4, 8], 4),
]


def _cu(doc_lengths, device):
    return torch.tensor([0, *torch.tensor(doc_lengths).cumsum(0).tolist()],
                        dtype=torch.int32, device=device)


def _ref_perdoc(q, k, v, cu, scaling):
    """Ground truth: standard causal attention within each document, concatenated.
    q/k/v: [1, H, S, d] -> returns [1, S, H, d]."""
    outs = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        od = F.scaled_dot_product_attention(
            q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, scale=scaling
        )
        outs.append(od)
    return torch.cat(outs, dim=2).transpose(1, 2)


def _worker(rank, world, seq_len, doc_lengths, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)  # identical full tensors on every rank
        H, d = 4, 8
        q = torch.randn(1, H, seq_len, d, dtype=torch.float64)
        k = torch.randn(1, H, seq_len, d, dtype=torch.float64)
        v = torch.randn(1, H, seq_len, d, dtype=torch.float64)
        scaling = 1.0 / (d ** 0.5)
        cu = _cu(doc_lengths, q.device)

        ref = _ref_perdoc(q, k, v, cu, scaling)  # [1, S, H, d]

        L = seq_len // world
        sl = slice(rank * L, (rank + 1) * L)
        out = varlen_ring_attention(
            q[:, :, sl], k[:, :, sl], v[:, :, sl],
            group=dist.group.WORLD, scaling=scaling, cu_seqlens=cu,
        )  # [1, L, H, d]
        err = (out - ref[:, sl]).abs().max().item()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seq_len,doc_lengths,world", CASES)
def test_varlen_ring_matches_perdoc(seq_len, doc_lengths, world):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, seq_len, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=90) for _ in range(world)]
    for p in procs:
        p.join(timeout=90)
    for rank, err in results:
        assert err < 1e-9, f"rank {rank} varlen ring vs per-doc ref err {err} (docs={doc_lengths})"


def _route_worker(rank, world, backend_name, lb_name, doc_lengths, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from types import SimpleNamespace

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        seq = sum(doc_lengths)
        H, d = 8, 8
        torch.manual_seed(0)
        q = torch.randn(1, H, seq, d, dtype=torch.float64)
        k = torch.randn(1, H, seq, d, dtype=torch.float64)
        v = torch.randn(1, H, seq, d, dtype=torch.float64)
        scaling = 1.0 / (d ** 0.5)
        cu = _cu(doc_lengths, q.device)
        ref = _ref_perdoc(q, k, v, cu, scaling)  # [1, S, H, d]

        cfg = dict(size=world, backend=rm.Backend(backend_name),
                   load_balance=rm.LoadBalance(lb_name))
        if backend_name == "usp":
            cfg.update(ulysses_size=2, ring_size=world // 2)
        runtime = rm.setup(rm.RingmasterConfig(**cfg), num_kv_heads=H,
                           device_mesh=None, inner_attn="flash_attention_2")
        runtime.varlen = (cu, int(cu.diff().max()))

        fn = ALL_ATTENTION_FUNCTIONS.get(runtime.attn_implementation)
        module = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=runtime.attn_implementation)
        )
        L = seq // world
        sl = slice(rank * L, (rank + 1) * L)
        out, _ = fn(module, q[:, :, sl], k[:, :, sl], v[:, :, sl], None,
                    scaling=scaling, is_causal=True)  # [1, L, H, d]
        err = (out - ref[:, sl]).abs().max().item()
        rm.teardown()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


# Backend dispatch: ring (none/head_tail/distflash all downgrade to masked ring under
# packing) + USP (2x2). Proves every backend's forward routes packing to varlen_ring.
ROUTE_CASES = [
    ("ring", "none", [6, 10], 2),
    ("ring", "distflash", [5, 7, 4], 2),   # genuine distflash schedule, doc-masked
    ("usp", "none", [10, 9, 13], 4),       # 2x2 ulysses x ring
]


@pytest.mark.parametrize("backend,lb,doc_lengths,world", ROUTE_CASES)
def test_varlen_backend_dispatch(backend, lb, doc_lengths, world):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_route_worker, args=(r, world, backend, lb, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=90) for _ in range(world)]
    for p in procs:
        p.join(timeout=90)
    # distflash does several online-softmax merges (fp64 ~1e-7); plain ring is one SDPA
    for rank, err in results:
        assert err < 1e-6, f"rank {rank} {backend}/{lb} varlen dispatch err {err}"


def _distflash_bwd_worker(rank, world, seq_len, doc_lengths, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29614")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from ringmaster.ring.distflash import distflash_attention

        torch.manual_seed(0)
        H, d = 4, 8
        scaling = 1.0 / (d ** 0.5)
        full = {n: torch.randn(1, H, seq_len, d, dtype=torch.float64) for n in "qkv"}
        cu = _cu(doc_lengths, full["q"].device)
        grad = torch.randn(1, seq_len, H, d, dtype=torch.float64)

        # reference: autograd through per-document causal attention
        qf = full["q"].clone().requires_grad_(True)
        kf = full["k"].clone().requires_grad_(True)
        vf = full["v"].clone().requires_grad_(True)
        _ref_perdoc(qf, kf, vf, cu, scaling).backward(grad)

        L = seq_len // world
        sl = slice(rank * L, (rank + 1) * L)
        qs = full["q"][:, :, sl].clone().requires_grad_(True)
        ks = full["k"][:, :, sl].clone().requires_grad_(True)
        vs = full["v"][:, :, sl].clone().requires_grad_(True)
        out = distflash_attention(qs, ks, vs, group=dist.group.WORLD, scaling=scaling,
                                  cu_seqlens=cu)
        out.backward(grad[:, sl])
        err = max(
            (qs.grad - qf.grad[:, :, sl]).abs().max().item(),
            (ks.grad - kf.grad[:, :, sl]).abs().max().item(),
            (vs.grad - vf.grad[:, :, sl]).abs().max().item(),
        )
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seq_len,doc_lengths,world", [
    (16, [5, 7, 4], 2),
    (16, [6, 10], 2),
    (16, [3, 9, 4], 4),
])
def test_distflash_varlen_backward(seq_len, doc_lengths, world):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_distflash_bwd_worker, args=(r, world, seq_len, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=120)
    for rank, err in results:
        assert err < 1e-6, f"rank {rank} distflash varlen bwd err {err} (docs={doc_lengths})"


def _zigzag_bwd_worker(rank, world, seq_len, doc_lengths, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29615")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from ringmaster.ring.zigzag import _zigzag_gidx, zigzag_ring_attention

        torch.manual_seed(0)
        H, d = 4, 8
        scaling = 1.0 / (d ** 0.5)
        full = {n: torch.randn(1, H, seq_len, d, dtype=torch.float64) for n in "qkv"}
        cu = _cu(doc_lengths, full["q"].device)
        grad = torch.randn(1, seq_len, H, d, dtype=torch.float64)

        qf = full["q"].clone().requires_grad_(True)
        kf = full["k"].clone().requires_grad_(True)
        vf = full["v"].clone().requires_grad_(True)
        ref = _ref_perdoc(qf, kf, vf, cu, scaling)  # [1, S, H, d]
        ref.backward(grad)

        half = seq_len // (2 * world)
        gidx = _zigzag_gidx(rank, half, world, full["q"].device)  # this rank's global positions
        qs = full["q"][:, :, gidx, :].clone().requires_grad_(True)
        ks = full["k"][:, :, gidx, :].clone().requires_grad_(True)
        vs = full["v"][:, :, gidx, :].clone().requires_grad_(True)
        out = zigzag_ring_attention(qs, ks, vs, group=dist.group.WORLD, scaling=scaling,
                                    cu_seqlens=cu)  # [1, 2*half, H, d]
        out.backward(grad[:, gidx])
        err = max(
            (out.detach() - ref.detach()[:, gidx]).abs().max().item(),
            (qs.grad - qf.grad[:, :, gidx, :]).abs().max().item(),
            (ks.grad - kf.grad[:, :, gidx, :]).abs().max().item(),
            (vs.grad - vf.grad[:, :, gidx, :]).abs().max().item(),
        )
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seq_len,doc_lengths,world", [
    (16, [6, 10], 2),
    (16, [5, 7, 4], 2),
    (32, [10, 9, 13], 4),
])
def test_zigzag_varlen_fwd_bwd(seq_len, doc_lengths, world):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_zigzag_bwd_worker, args=(r, world, seq_len, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=120)
    for rank, err in results:
        assert err < 1e-6, f"rank {rank} zigzag varlen fwd/bwd err {err} (docs={doc_lengths})"


def test_cu_seqlens_and_varlen_meta():
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4]])  # docs 4, 2, 5
    cu, ml = cu_seqlens_from_position_ids(pos)
    assert cu.tolist() == [0, 4, 6, 11] and ml == 5
    assert cu_seqlens_from_position_ids(torch.arange(8).view(1, 8)) is None  # dense
    cu2, ml2 = varlen_meta(pos, 16)  # pad 11 -> 16 becomes a trailing segment
    assert cu2.tolist() == [0, 4, 6, 11, 16] and ml2 == 5
    assert varlen_meta(torch.arange(8).view(1, 8), 8) is None
