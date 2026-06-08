"""Gated multi-GPU test: DistFlashAttn-style balanced contiguous ring — full
fwd+bwd vs single-GPU causal attention at P=2 (NCCL). Validates the hand-written
backward on the real target (the CPU/gloo backward deadlocks on autograd worker
threads, so CPU only validates the forward schedule). Requires >= 2 CUDA devices."""

import os

import pytest
import torch

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _worker(rank, world, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29658")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from ringmaster.ring.distflash import distflash_attention

        dev = torch.device("cuda", rank)
        B, H, D, S = 1, 8, 64, 1024
        loc = S // world
        torch.manual_seed(0)
        gq = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        gk = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        gv = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
        for t in (gq, gk, gv):
            dist.broadcast(t, 0)
        sl = slice(rank * loc, (rank + 1) * loc)
        q = gq[:, :, sl].clone().requires_grad_()
        k = gk[:, :, sl].clone().requires_grad_()
        v = gv[:, :, sl].clone().requires_grad_()
        out = distflash_attention(q, k, v, group=dist.group.WORLD, scaling=1 / D ** 0.5)
        (out.float() ** 2).sum().backward()

        def gath(x, seqdim):
            xs = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(xs, x.contiguous())
            return torch.cat(xs, dim=seqdim)

        o = gath(out.float(), 1)
        dq = gath(q.grad.float(), 2)
        dk = gath(k.grad.float(), 2)
        dv = gath(v.grad.float(), 2)
        errs = None
        if rank == 0:
            qr = gq.float().clone().requires_grad_()
            kr = gk.float().clone().requires_grad_()
            vr = gv.float().clone().requires_grad_()
            s = (qr @ kr.transpose(-1, -2)) * (1 / D ** 0.5)
            m = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
            ro = (s.masked_fill(~m, float("-inf")).softmax(-1) @ vr).transpose(1, 2)
            (ro ** 2).sum().backward()
            def e(a, b):
                return ((a - b).abs().mean() / (b.abs().mean() + 1e-6)).item()
            errs = (e(o, ro), e(dq, qr.grad), e(dk, kr.grad), e(dv, vr.grad))
        out_q.put((rank, errs))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not CUDA_OK, reason="needs >= 2 CUDA devices")
def test_distflash_gpu_fwd_bwd_matches_full_causal():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, out_q)) for r in range(2)]
    for p in procs:
        p.start()
    res = dict(out_q.get(timeout=180) for _ in range(2))
    for p in procs:
        p.join(timeout=180)
    o, dq, dk, dv = res[0]
    assert o < 8e-3 and dq < 8e-3 and dk < 8e-3 and dv < 8e-3, f"errs out={o} dq={dq} dk={dk} dv={dv}"
