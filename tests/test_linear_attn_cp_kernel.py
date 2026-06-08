"""Gated multi-GPU test: linear-attention (gated DeltaNet) CP via fla's native cp_context.

Qwen3.5 / Qwen3-Next linear-attention layers use flash-linear-attention's gated
delta rule. fla (>=0.5.1) ships native context parallelism: build_cp_context +
cp_context on the kernel. The sharded result must match the full-sequence kernel.
Requires >= 2 CUDA devices + flash-linear-attention with cp_context.
"""

import os

import pytest
import torch

try:
    import inspect

    from fla.ops.cp import build_cp_context  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA_CP = "cp_context" in inspect.signature(chunk_gated_delta_rule).parameters
except Exception:
    HAVE_FLA_CP = False

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _worker(rank, world, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29609")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        torch.manual_seed(0)
        B, T, H, K, Vd = 1, 256, 8, 128, 128

        def mk(s):
            t = torch.randn(*s, device=rank, dtype=torch.bfloat16)
            dist.broadcast(t, 0)
            return t

        q, k, v = mk((B, T, H, K)), mk((B, T, H, K)), mk((B, T, H, Vd))
        g = -torch.nn.functional.softplus(torch.randn(B, T, H, device=rank, dtype=torch.float32))
        dist.broadcast(g, 0)
        beta = torch.rand(B, T, H, device=rank, dtype=torch.bfloat16).sigmoid()
        dist.broadcast(beta, 0)

        cu = torch.tensor([0, T], device=rank, dtype=torch.long)
        ref, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=True
        )
        ctx = build_cp_context(cu, group=dist.group.WORLD)
        h = T // world
        sl = slice(rank * h, (rank + 1) * h)
        o, _ = chunk_gated_delta_rule(
            q=q[:, sl], k=k[:, sl], v=v[:, sl], g=g[:, sl], beta=beta[:, sl],
            cp_context=ctx, use_qk_l2norm_in_kernel=True,
        )
        err = (o.float() - ref[:, sl].float()).abs().max().item() / (
            ref.float().abs().max().item() + 1e-6
        )
        out_q.put((rank, err))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (CUDA_OK and HAVE_FLA_CP),
    reason="needs >=2 CUDA devices + flash-linear-attention with cp_context (>=0.5.1)",
)
def test_gated_delta_native_cp_matches_full():
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
        assert err < 1e-2, f"rank {rank} native gated-delta CP rel err {err}"
