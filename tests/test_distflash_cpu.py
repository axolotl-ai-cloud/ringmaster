"""DistFlash schedule correctness on CPU/gloo at P=2,4,8 — FORWARD and BACKWARD vs
full causal attention, covering the P>=4 sizes the 2-GPU box can't run.

Backward is CPU-validatable because the phase-1 routing is symmetric (matched
send+recv each step); asymmetric routing used to deadlock gloo on the autograd worker
thread. Full fwd+bwd also runs on GPU/NCCL at P=2 (tests/test_distflash_gpu.py).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ringmaster.ring.distflash import distflash_attention
from ringmaster.ring.kernels import math_block


def _fwd_worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29657"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        b, h, total, d = 1, 4, 8 * world, 16
        qf = torch.randn(b, h, total, d)
        kf = torch.randn(b, h, total, d)
        vf = torch.randn(b, h, total, d)
        ref, _ = math_block(qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2),
                            causal=True, scaling=None)  # [b,total,h,d]
        local = total // world
        sl = slice(rank * local, (rank + 1) * local)
        with torch.no_grad():
            out = distflash_attention(qf[:, :, sl], kf[:, :, sl], vf[:, :, sl],
                                      group=dist.group.WORLD, scaling=None)
        out_q.put((rank, (out - ref[:, sl]).abs().max().item()))
    finally:
        dist.destroy_process_group()


def _bwd_worker(rank, world, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29659"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        b, h, total, d = 1, 4, 8 * world, 16
        gq = torch.randn(b, h, total, d, dtype=torch.float64)
        gk = torch.randn(b, h, total, d, dtype=torch.float64)
        gv = torch.randn(b, h, total, d, dtype=torch.float64)
        local = total // world
        sl = slice(rank * local, (rank + 1) * local)

        q = gq[:, :, sl].clone().requires_grad_()
        k = gk[:, :, sl].clone().requires_grad_()
        v = gv[:, :, sl].clone().requires_grad_()
        out = distflash_attention(q, k, v, group=dist.group.WORLD, scaling=None)
        (out ** 2).sum().backward()  # global loss = sum_r (out_r**2).sum()

        def gath(x, seqdim):
            xs = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(xs, x.contiguous())
            return torch.cat(xs, dim=seqdim)

        dq = gath(q.grad, 2)  # [b,h,total,d]
        dk = gath(k.grad, 2)
        dv = gath(v.grad, 2)
        o = gath(out, 1)      # [b,total,h,d]
        errs = None
        if rank == 0:
            qr = gq.clone().requires_grad_()  # [b,h,total,d]
            kr = gk.clone().requires_grad_()
            vr = gv.clone().requires_grad_()
            sc = (qr @ kr.transpose(-1, -2)) * (1.0 / d ** 0.5)  # [b,h,total,total]
            m = torch.ones(total, total, dtype=torch.bool).tril()
            ro = sc.masked_fill(~m, float("-inf")).softmax(-1) @ vr  # [b,h,total,d]
            (ro ** 2).sum().backward()  # global loss = sum_r (out_r**2).sum()

            def e(a, b):
                return (a - b).abs().max().item()

            errs = (e(o, ro.transpose(1, 2).detach()), e(dq, qr.grad),
                    e(dk, kr.grad), e(dv, vr.grad))
        out_q.put((rank, errs))
    finally:
        dist.destroy_process_group()


def _run(target, world):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    res = [out_q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=120)
    return res


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 4, 8])
def test_distflash_forward_matches_full_causal(world):
    for rank, err in _run(_fwd_worker, world):
        assert err < 1e-4, f"world={world} rank={rank} fwd err={err}"


@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 4, 8])
def test_distflash_backward_matches_full_causal(world):
    for rank, errs in _run(_bwd_worker, world):
        if errs is None:
            continue
        o, dq, dk, dv = errs  # explicit fallback computes in fp32 (.float()) -> ~1e-6
        assert o < 1e-4 and dq < 1e-4 and dk < 1e-4 and dv < 1e-4, \
            f"world={world} errs out={o} dq={dq} dk={dk} dv={dv}"
