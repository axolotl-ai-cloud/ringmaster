"""Float32 reference control for Mamba1 state passing and cross-rank gradients."""

import torch
import torch.distributed as dist

from ringmaster.config import RingmasterConfig
from ringmaster.fla_mamba import _scan1
from ringmaster.runtime import CPRuntime, set_runtime


def reference(u, dt, A, B, C, D, z, bias, **kwargs):
    effective = torch.nn.functional.softplus(dt + bias[None, :, None])
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[-1])
    outputs = []
    for t in range(u.shape[-1]):
        state = (effective[:, :, t, None] * A).exp() * state + effective[
            :, :, t, None
        ] * u[:, :, t, None] * B[:, None, :, t]
        outputs.append(
            ((state * C[:, None, :, t]).sum(-1) + D * u[:, :, t])
            * torch.nn.functional.silu(z[:, :, t])
        )
    return torch.stack(outputs, -1)


def main():
    dist.init_process_group("gloo")
    global_rank = dist.get_rank()
    group = dist.group.WORLD
    if dist.get_world_size() == 8:
        for ranks in (list(range(0, 8, 2)), list(range(1, 8, 2))):
            candidate = dist.new_group(ranks)
            if global_rank in ranks:
                group = candidate
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    try:
        for packed in (False, True):
            torch.manual_seed(4 + global_rank % 2 if dist.get_world_size() == 8 else 4)
            tensors = [
                torch.randn(2, 3, 24),
                torch.randn(2, 3, 24),
                -torch.rand(3, 2),
                torch.randn(2, 2, 24),
                torch.randn(2, 2, 24),
                torch.randn(3),
                torch.randn(2, 3, 24),
                torch.randn(3),
            ]
            tensors = [t.requires_grad_() for t in tensors]
            rows = []
            boundaries = (
                ((0, 5, 16, 24), (0, 7, 9, 24)) if packed else ((0, 24), (0, 24))
            )
            for row, starts in enumerate(boundaries):
                rows.append(
                    torch.cat(
                        [
                            reference(
                                *(
                                    t[row : row + 1, :, a:b]
                                    if i in (0, 1, 3, 4, 6)
                                    else t
                                    for i, t in enumerate(tensors)
                                )
                            )
                            for a, b in zip(starts[:-1], starts[1:])
                        ],
                        -1,
                    )
                )
            expected = torch.cat(rows)
            expected.square().sum().backward()
            grads = [t.grad.clone() for t in tensors]
            start, end = rank * 24 // world, (rank + 1) * 24 // world
            local = [
                (t[:, :, start:end] if i in (0, 1, 3, 4, 6) else t)
                .detach()
                .clone()
                .requires_grad_()
                for i, t in enumerate(tensors)
            ]
            config = RingmasterConfig(size=world)
            runtime = CPRuntime(config=config, cp_group=group)
            if packed:
                runtime.varlen = (
                    torch.tensor([0, 5, 16, 24, 31, 33, 48], dtype=torch.int32),
                    15,
                )
            set_runtime(runtime)
            actual = _scan1(reference, group)(*local)
            torch.testing.assert_close(
                actual, expected[:, :, start:end], atol=2e-5, rtol=2e-5
            )
            actual.square().sum().backward()
            for i, tensor in enumerate(local):
                if i in (0, 1, 3, 4, 6):
                    expected_grad = grads[i][:, :, start:end]
                else:
                    dist.all_reduce(tensor.grad, group=group)
                    expected_grad = grads[i]
                torch.testing.assert_close(
                    tensor.grad, expected_grad, atol=5e-5, rtol=5e-5
                )
            print(f"PASS rank={global_rank} packed={packed}", flush=True)
    finally:
        set_runtime(None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
