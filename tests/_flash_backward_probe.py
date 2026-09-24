"""Random-input GQA accuracy across the dense Ring backward implementations."""

import os

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ringmaster.config import RotateMethod
from ringmaster.ring.distflash import distflash_attention
from ringmaster.ring.loop import ring_attention
from ringmaster.ring.zigzag import zigzag_ring_attention


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        world = dist.get_world_size()
        for mode in ("allgather", "head_tail", "distflash"):
            torch.manual_seed(29)
            q = torch.randn(1, 4, 128, 32, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 2, 128, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            ref = [t.clone().requires_grad_() for t in (q, k, v)]
            expected = F.scaled_dot_product_attention(*ref, is_causal=True, enable_gqa=True).transpose(1, 2)
            grad = torch.randn_like(expected)
            (expected * grad).sum().backward()
            if mode == "head_tail":
                parts = torch.arange(128, device="cuda").chunk(2 * world)
                indices = torch.cat((parts[rank], parts[2 * world - 1 - rank]))
            else:
                indices = torch.arange(128, device="cuda").chunk(world)[rank]
            local = [t[:, :, indices].contiguous().requires_grad_() for t in (q, k, v)]
            if mode == "allgather":
                actual = ring_attention(*local, group=dist.group.WORLD, causal=True, scaling=None,
                                        dropout=0.0, provider="hf_kernels", rotate_method=RotateMethod.ALLGATHER,
                                        attn_implementation="flash_attention_2")
            else:
                kernel = zigzag_ring_attention if mode == "head_tail" else distflash_attention
                actual = kernel(*local, group=dist.group.WORLD, scaling=None)
            torch.testing.assert_close(actual, expected[:, indices], atol=0.015, rtol=0.04)
            (actual * grad[:, indices]).sum().backward()
            errors = []
            for tensor, target in zip(local, ref, strict=True):
                target_grad = target.grad[:, :, indices]
                errors.append((tensor.grad - target_grad).abs().max().item())
                torch.testing.assert_close(tensor.grad, target_grad, atol=0.03, rtol=0.05)
            print(f"PASS {mode} rank={rank} GQA gradient_errors={errors}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
