"""CPU/gloo USP (2x2) parity via the math block kernel.

Validates the hybrid Ulysses x Ring composition end-to-end on 4 ranks (which a
2-GPU box can't do with NCCL): the gathered USP output must equal full attention.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29603")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster.config import RotateMethod
        from ringmaster.ring.kernels import math_block
        from ringmaster.strategies.usp import make_usp_attention

        rm.setup(
            rm.RingmasterConfig(
                size=world, backend=rm.Backend.USP, ulysses_size=2, ring_size=2
            ),
            num_kv_heads=4,
            device_mesh=None,
            device_type="cpu",
            inner_attn="math",
        )

        torch.manual_seed(0)  # identical full tensors on every rank
        b, h, total, d = 1, 4, 8, 16
        qf = torch.randn(b, h, total, d, dtype=torch.float64, requires_grad=True)
        kf = torch.randn(b, h, total, d, dtype=torch.float64, requires_grad=True)
        vf = torch.randn(b, h, total, d, dtype=torch.float64, requires_grad=True)

        ref, _ = math_block(
            qf.transpose(1, 2),
            kf.transpose(1, 2),
            vf.transpose(1, 2),
            causal=True,
            scaling=None,
        )  # [b, total, h, d]

        local = total // world
        sl = slice(rank * local, (rank + 1) * local)
        q = qf[:, :, sl].detach().contiguous().requires_grad_()
        k = kf[:, :, sl].detach().contiguous().requires_grad_()
        v = vf[:, :, sl].detach().contiguous().requires_grad_()

        usp_fwd = make_usp_attention("math", "math", RotateMethod.ALLGATHER)
        out, _ = usp_fwd(None, q, k, v, None, scaling=None, is_causal=True)
        grad = torch.randn_like(ref)
        expected_grads = torch.autograd.grad((ref * grad).sum(), (qf, kf, vf))
        actual_grads = torch.autograd.grad((out * grad[:, sl]).sum(), (q, k, v))
        for actual, expected in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual, expected[:, :, sl], atol=1e-6, rtol=1e-5)
        # out: [b, s_local, h, d]; gather across ranks to rebuild the full sequence
        gathered = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(gathered, out.contiguous(), group=dist.group.WORLD)
        full = torch.cat(gathered, dim=1)

        err = (full - ref).abs().max().item()
        out_q.put((rank, err))
        rm.teardown()
    finally:
        dist.destroy_process_group()


def test_usp_2x2_matches_full_attention():
    world = 4
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=90) for _ in range(world)]
    for p in procs:
        p.join(timeout=90)
    for rank, err in results:
        assert err < 1e-6, f"rank {rank} USP 2x2 vs full attention err {err}"
