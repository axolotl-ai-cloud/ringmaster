"""CPU/gloo correctness test for Mamba2 SSM context parallelism.

A reference global SSD recurrence vs. the CP scheme must match. Two paths: the
single-hop ``ring_shift_ssm_state`` (exact only at P=2) and the exact ``cp_state_prefix``
(all-gather + local prefix-combine; exact for any P) used by ``wrap_mamba_scan_for_cp``.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _ssd(x, dt, A, B, C):
    """Reference Mamba2 SSD: h_t = exp(A*dt_t)*h_{t-1} + (dt_t*B_t)⊗x_t; y_t = C_t·h_t.
    x:[b,T,H,d] dt:[b,T,H] A:[H] B,C:[b,T,G,n]. Returns out [b,T,H*d], h_final [b,H,d,n]."""
    bsz, T, H, d = x.shape
    G, n = B.shape[2], B.shape[3]
    hpg = H // G
    Bx = B.repeat_interleave(hpg, dim=2)  # [b,T,H,n]
    Cx = C.repeat_interleave(hpg, dim=2)
    h = torch.zeros(bsz, H, d, n, dtype=x.dtype)
    outs = []
    for t in range(T):
        dA = torch.exp(A * dt[:, t])  # [b,H]
        inp = dt[:, t][:, :, None, None] * x[:, t][:, :, :, None] * Bx[:, t][:, :, None, :]
        h = dA[:, :, None, None] * h + inp
        y = (Cx[:, t][:, :, None, :] * h).sum(-1)  # [b,H,d]
        outs.append(y.reshape(bsz, H * d))
    return torch.stack(outs, 1), h


def _worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29605")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from ringmaster.config import RingmasterConfig
        from ringmaster.runtime import CPRuntime, set_runtime
        from ringmaster.strategies.mamba import (
            mamba2_cp_correction,
            ring_shift_ssm_state,
        )

        set_runtime(CPRuntime(config=RingmasterConfig(size=world), cp_group=dist.group.WORLD))

        torch.manual_seed(0)  # identical full tensors on every rank
        b, T, H, d, G, n = 1, 8, 2, 3, 1, 4
        x = torch.randn(b, T, H, d, dtype=torch.float64)
        dt = torch.rand(b, T, H, dtype=torch.float64) * 0.1
        A = -torch.rand(H, dtype=torch.float64)
        B = torch.randn(b, T, G, n, dtype=torch.float64)
        C = torch.randn(b, T, G, n, dtype=torch.float64)

        ref_out, _ = _ssd(x, dt, A, B, C)  # [b,T,H*d]

        local = T // world
        sl = slice(rank * local, (rank + 1) * local)
        out_c, h_final = _ssd(x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl])  # zero-init
        cum_A = torch.cumsum(A[None, None, :] * dt[:, sl], dim=1)  # [b,Tc,H]

        h_prev = ring_shift_ssm_state(h_final)
        corrected, _ = mamba2_cp_correction(
            out_c, h_final, C[:, sl], cum_A, h_prev, num_heads=H, head_dim=d
        )
        err = (corrected - ref_out[:, sl]).abs().max().item()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


def test_mamba2_cp_correction_matches_global():
    world = 2  # single-hop shift is exact for 2 ranks
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=60) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)
    for rank, err in results:
        assert err < 1e-6, f"rank {rank} mamba CP vs global err {err}"


def _prefix_worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29607")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from ringmaster.config import RingmasterConfig
        from ringmaster.runtime import CPRuntime, set_runtime
        from ringmaster.strategies.mamba import cp_state_prefix, mamba2_cp_correction

        set_runtime(CPRuntime(config=RingmasterConfig(size=world), cp_group=dist.group.WORLD))
        torch.manual_seed(0)
        b, T, H, d, G, n = 1, 16, 2, 3, 1, 4
        x = torch.randn(b, T, H, d, dtype=torch.float64)
        dt = torch.rand(b, T, H, dtype=torch.float64) * 0.1
        A = -torch.rand(H, dtype=torch.float64)
        B = torch.randn(b, T, G, n, dtype=torch.float64)
        C = torch.randn(b, T, G, n, dtype=torch.float64)

        ref_out, _ = _ssd(x, dt, A, B, C)

        local = T // world
        sl = slice(rank * local, (rank + 1) * local)
        out_c, h_final = _ssd(x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl])  # B_r = h_final
        cum_A = torch.cumsum(A[None, None, :] * dt[:, sl], dim=1)  # [b, Tc, H]
        chunk_decay = torch.exp(cum_A[:, -1])  # A_r: [b, H]

        h_prev, _ = cp_state_prefix(h_final, chunk_decay, dist.group.WORLD)
        corrected, _ = mamba2_cp_correction(
            out_c, h_final, C[:, sl], cum_A, h_prev, num_heads=H, head_dim=d
        )
        err = (corrected - ref_out[:, sl]).abs().max().item()
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_mamba2_cp_prefix_matches_global(world):
    """The exact prefix-combine path must match the global SSD for any P (not just 2)."""
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_prefix_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=90) for _ in range(world)]
    for p in procs:
        p.join(timeout=90)
    for rank, err in results:
        assert err < 1e-6, f"rank {rank}/{world} mamba prefix vs global err {err}"
