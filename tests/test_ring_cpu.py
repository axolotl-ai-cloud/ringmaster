"""CPU/gloo ring parity (forward + backward) via the math block kernel.

Validates the ring loop and the autograd-aware p2p path without GPUs/flash:
gathered ring output and per-shard gradients must match full attention.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ringmaster.config import RotateMethod
from ringmaster.ring import ring_attention
from ringmaster.ring.kernels import math_block


def _worker(rank, world, rotate, train, out_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)  # identical full tensors on every rank
        b, h, total, d = 1, 4, 8, 16
        qf = torch.randn(b, h, total, d, dtype=torch.float64)
        kf = torch.randn(b, h, total, d, dtype=torch.float64)
        vf = torch.randn(b, h, total, d, dtype=torch.float64)

        qf_ref = qf.clone().requires_grad_(train)
        kf_ref = kf.clone().requires_grad_(train)
        out_full, _ = math_block(
            qf_ref.transpose(1, 2),
            kf_ref.transpose(1, 2),
            vf.transpose(1, 2),
            causal=True,
            scaling=None,
        )  # [b, s, h, d]
        if train:
            out_full.sum().backward()

        local = total // world
        sl = slice(rank * local, (rank + 1) * local)
        q = qf[:, :, sl].clone().requires_grad_(train)
        k = kf[:, :, sl].clone().requires_grad_(train)
        v = vf[:, :, sl].clone()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            out = ring_attention(
                q, k, v,
                group=dist.group.WORLD,
                causal=True,
                scaling=None,
                dropout=0.0,
                provider="math",
                rotate_method=rotate,
                attn_implementation="math",
            )  # [b, s_local, h, d]

        fwd_err = (out - out_full[:, sl]).abs().max().item()
        if train:
            out.sum().backward()  # collective backward across ranks
            q_err = (q.grad - qf_ref.grad[:, :, sl]).abs().max().item()
            k_err = (k.grad - kf_ref.grad[:, :, sl]).abs().max().item()
        else:
            q_err = k_err = 0.0
        out_q.put((rank, fwd_err, q_err, k_err))
    finally:
        dist.destroy_process_group()


def _run(rotate, train):
    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, rotate, train, out_q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    res = [out_q.get(timeout=60) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)
    return res


def test_ring_allgather_forward_and_backward():
    # allgather is the training path: forward + gradients must match full attention.
    for rank, fwd, qg, kg in _run(RotateMethod.ALLGATHER, train=True):
        assert fwd < 1e-6, f"rank {rank} fwd {fwd}"
        assert qg < 1e-6 and kg < 1e-6, f"rank {rank} grad q={qg} k={kg}"


def test_ring_p2p_forward_and_backward():
    # p2p (memory-optimal) ring: forward + gradients must match full attention.
    for rank, fwd, qg, kg in _run(RotateMethod.ALLTOALL, train=True):
        assert fwd < 1e-6, f"rank {rank} fwd {fwd}"
        assert qg < 1e-6 and kg < 1e-6, f"rank {rank} grad q={qg} k={kg}"
