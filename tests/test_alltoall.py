"""Multi-process (gloo/CPU) test for the Ulysses all-to-all round-trip."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ringmaster.comm import seq_all_to_all


def _worker(rank: int, world: int, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        b, heads, s_local, d = 1, world, 2, 3
        # Each rank holds [b, H, s/P, d]; fill so heads are globally distinguishable.
        x = torch.arange(b * heads * s_local * d, dtype=torch.float32).reshape(
            b, heads, s_local, d
        ) + rank * 1000
        x.requires_grad_(True)

        # [b,H,s/P,d] -> [b,H/P,S,d] (scatter heads, gather seq)
        gathered = seq_all_to_all(x, scatter_dim=1, gather_dim=2, group=group)
        assert tuple(gathered.shape) == (b, heads // world, s_local * world, d)

        # inverse -> back to original
        restored = seq_all_to_all(gathered, scatter_dim=2, gather_dim=1, group=group)
        assert tuple(restored.shape) == tuple(x.shape)
        roundtrip_ok = torch.allclose(restored, x.detach())

        # autograd: grad of sum() w.r.t. x is all ones
        gathered.sum().backward()
        grad_ok = torch.allclose(x.grad, torch.ones_like(x))

        out_q.put((rank, bool(roundtrip_ok), bool(grad_ok)))
    finally:
        dist.destroy_process_group()


def test_ulysses_all_to_all_roundtrip_and_grad():
    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, out_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=55) for _ in range(world)]
    for p in procs:
        p.join(timeout=55)

    assert len(results) == world
    for _rank, roundtrip_ok, grad_ok in results:
        assert roundtrip_ok, "all-to-all round trip mismatch"
        assert grad_ok, "all-to-all backward mismatch"
