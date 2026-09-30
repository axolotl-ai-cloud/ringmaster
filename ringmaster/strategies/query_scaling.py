"""Restore global query positions for layers that scale before attention dispatch."""

import torch

from ringmaster.runtime import get_runtime


def global_query_scale(module, query):
    if not (
        getattr(module, "attn_temperature_tuning", False)
        and not getattr(module, "use_rope", True)
    ):
        return query
    runtime = get_runtime()
    length = query.shape[-2]
    local = torch.arange(length, device=query.device, dtype=torch.float32)
    if runtime.shard_load_balance == "head_tail":
        half = length // 2
        global_positions = torch.cat(
            (
                local[:half] + runtime.cp_rank * half,
                local[half:]
                + (2 * runtime.cp_size - 1 - runtime.cp_rank) * half
                - half,
            )
        )
    else:
        global_positions = local + runtime.cp_rank * length
    floor_scale = module.floor_scale
    attn_scale = module.attn_scale

    def scale(positions):
        return 1 + torch.log1p(torch.floor((positions + 1) / floor_scale)) * attn_scale

    ratio = scale(global_positions) / scale(local)
    return query * ratio.to(query.dtype)[None, None, :, None]
