"""Multi-process (gloo/CPU) test: CP linear scan == global sequential scan.

Validates the exact state-passing primitive used for Mamba2 / linear-attention CP.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ringmaster.strategies.state_passing import cp_linear_scan, local_linear_scan


def _worker(rank: int, world: int, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)  # identical full sequence on every rank
        batch, total_len, d = 1, 8, 3
        a_full = torch.rand(batch, total_len, d) * 0.9 + 0.05  # decays in (0,1)
        b_full = torch.randn(batch, total_len, d)

        # global reference
        ref = local_linear_scan(a_full, b_full)

        # this rank's contiguous chunk
        local = total_len // world
        sl = slice(rank * local, (rank + 1) * local)
        got = cp_linear_scan(a_full[:, sl], b_full[:, sl], dist.group.WORLD)

        err = (got - ref[:, sl]).abs().max().item()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


def test_cp_linear_scan_matches_global():
    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=60) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)

    for rank, err in results:
        assert err < 1e-5, f"rank {rank} CP scan vs global scan err {err}"


def _gated_linear_attention_global(q, k, v, g):
    """Reference gated linear attention: S_t = g_t*S_{t-1} + k_t v_t^T; o_t = q_t·S_t.
    q,k,v: [T, dk]/[T, dv]; g: [T] gate. Returns o: [T, dv]."""
    dk, dv = k.shape[1], v.shape[1]
    S = torch.zeros(dk, dv, dtype=q.dtype)
    out = []
    for t in range(q.shape[0]):
        S = g[t] * S + torch.outer(k[t], v[t])
        out.append(q[t] @ S)
    return torch.stack(out, 0)


def _gla_worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29600")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(1)
        T, dk, dv = 8, 4, 5
        q = torch.randn(T, dk, dtype=torch.float64)
        k = torch.randn(T, dk, dtype=torch.float64)
        v = torch.randn(T, dv, dtype=torch.float64)
        g = torch.rand(T, dtype=torch.float64) * 0.9 + 0.05

        ref = _gated_linear_attention_global(q, k, v, g)

        # Cast the gated state recurrence to cp_linear_scan: state flattened to dk*dv,
        # decay = gate broadcast, input = (k outer v) flattened.
        local = T // world
        sl = slice(rank * local, (rank + 1) * local)
        kv = torch.einsum("ti,tj->tij", k[sl], v[sl]).reshape(1, local, dk * dv)
        a = g[sl].reshape(1, local, 1).expand(1, local, dk * dv)
        states = cp_linear_scan(a, kv, dist.group.WORLD).reshape(local, dk, dv)
        out = torch.einsum("ti,tij->tj", q[sl], states)

        err = (out - ref[sl]).abs().max().item()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


def test_cp_gated_linear_attention_matches_global():
    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_gla_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=60) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)
    for rank, err in results:
        assert err < 1e-5, f"rank {rank} CP gated-linear-attn vs global err {err}"
