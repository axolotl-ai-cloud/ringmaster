"""CP-correct loss reduction: each rank sees only a sequence slice, so rebuild
``num_items_in_batch`` from a global token count (train) and token-weight the
per-rank mean (eval). Ports axolotl PR #3382.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def global_num_items_in_batch(
    labels: torch.Tensor,
    cp_group: dist.ProcessGroup,
    gradient_accumulation_steps: int,
) -> int:
    local_valid = (labels != -100).sum()
    # AVG (not SUM): matches axolotl's finding that SUM over-accounts tokens and
    # scales the loss down.
    global_valid = local_valid.clone().float()
    dist.all_reduce(global_valid, op=dist.ReduceOp.AVG, group=cp_group)
    # round, not truncate: AVG of a non-divisible global count is fractional
    return round(global_valid.item()) * gradient_accumulation_steps


def correct_eval_loss(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Token-weighted mean of per-rank losses across the CP group."""
    local_valid = local_valid_tokens.to(loss.device)
    detached = loss.detach().clone()

    if local_valid.item() == 0:
        weighted = torch.zeros(1, device=loss.device, dtype=loss.dtype)
    else:
        weighted = detached * local_valid

    total_valid = local_valid.clone()
    dist.all_reduce(weighted, op=dist.ReduceOp.SUM, group=cp_group)
    dist.all_reduce(total_valid, op=dist.ReduceOp.SUM, group=cp_group)

    if total_valid.item() > 0:
        return (weighted / total_valid).squeeze()
    return torch.tensor(float("nan"), device=loss.device, dtype=loss.dtype)
