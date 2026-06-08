"""Gated multi-GPU test: Mamba2 CP with the REAL mamba-ssm kernel.

wrap_mamba_scan_for_cp + ring state-passing on sequence-sharded chunks must match
the full-sequence mamba_chunk_scan_combined. Requires >= 2 CUDA devices + mamba-ssm.
"""

import os

import pytest
import torch

try:
    import mamba_ssm.ops.triton.ssd_combined as _ssd  # noqa: F401

    HAVE_MAMBA = True
except Exception:
    HAVE_MAMBA = False

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _worker(rank, world, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29607")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        import mamba_ssm.ops.triton.ssd_combined as ssd

        from ringmaster.config import RingmasterConfig
        from ringmaster.runtime import CPRuntime, set_runtime
        from ringmaster.strategies.mamba import wrap_mamba_scan_for_cp

        set_runtime(CPRuntime(config=RingmasterConfig(size=world), cp_group=dist.group.WORLD))

        torch.manual_seed(0)  # identical full tensors on every rank
        b, T, H, d, n, G = 1, 64, 4, 32, 16, 1
        cs = 16
        x = torch.randn(b, T, H, d, device=rank, dtype=torch.bfloat16)
        dt = (torch.rand(b, T, H, device=rank, dtype=torch.bfloat16) * 0.1)
        A = -torch.rand(H, device=rank, dtype=torch.float32)
        B = torch.randn(b, T, G, n, device=rank, dtype=torch.bfloat16)
        C = torch.randn(b, T, G, n, device=rank, dtype=torch.bfloat16)

        orig = ssd.mamba_chunk_scan_combined
        with torch.no_grad():
            full = orig(x, dt, A, B, C, chunk_size=cs)  # [b,T,H,d]

        wrap_mamba_scan_for_cp(ssd)  # patches ssd.mamba_chunk_scan_combined

        local = T // world
        sl = slice(rank * local, (rank + 1) * local)
        with torch.no_grad():
            out, _state = ssd.mamba_chunk_scan_combined(
                x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl], chunk_size=cs
            )

        ref = full[:, sl].reshape(b, local, -1).float()
        got = out.reshape(b, local, -1).float()
        err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        out_q.put((rank, err))
        ssd.mamba_chunk_scan_combined = orig
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not (CUDA_OK and HAVE_MAMBA), reason="needs >=2 CUDA devices + mamba-ssm")
def test_mamba2_cp_kernel_matches_full():
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=180) for _ in range(world)]
    for p in procs:
        p.join(timeout=180)
    for rank, err in results:
        assert err < 5e-2, f"rank {rank} mamba2 CP (real kernel) rel err {err}"
