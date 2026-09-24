"""State-passing CP for recurrent mixers (Mamba/Mamba2, gated linear attention).

A linear recurrence ``h_t = a_t*h_{t-1}+b_t`` is linear in its initial state, so each
rank scans its chunk with ``h0=0``, all-gathers each chunk's (decay ``A_r``, carry
``B_r``) [O(P), seq-independent], prefix-combines previous ranks' (A,B) into the
entering state ``H_in``, and applies the exact correction ``h_t += cumA_t * H_in``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from ringmaster.recurrent import is_recurrent_mixer


def local_linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential reference scan h_t = a_t*h_{t-1}+b_t, zero init. a,b: [batch, L, d]."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def cp_linear_scan(a: torch.Tensor, b: torch.Tensor, group) -> torch.Tensor:
    """Exact context-parallel linear scan; matches a global scan of the full sequence.

    a, b: this rank's contiguous chunk, shape [batch, L_local, d]. Returns the
    corrected local hidden states [batch, L_local, d]. Comm: one all-gather of two
    [batch, d] vectors (O(P)), independent of sequence length.
    """
    import torch.distributed as dist

    local_h = local_linear_scan(a, b)
    cum_a = torch.cumprod(a, dim=1)  # [batch, L, d]

    world = dist.get_world_size(group) if group is not None else 1
    if world == 1:
        return local_h

    rank = dist.get_rank(group)
    chunk_decay = cum_a[:, -1]  # A_r: decay product over this chunk
    chunk_carry = local_h[:, -1]  # B_r: zero-init carry-out

    stacked = torch.stack([chunk_decay, chunk_carry], dim=0).contiguous()  # [2, b, d]
    from ringmaster.cp_collectives import all_gather_cp

    gathered = all_gather_cp(stacked, group)

    # Prefix-combine previous ranks' (A, B) to get the state entering this chunk.
    h_in = torch.zeros_like(chunk_carry)
    for j in range(rank):
        h_in = gathered[j][0] * h_in + gathered[j][1]

    return local_h + cum_a * h_in.unsqueeze(1) + gathered.sum() * 0


@dataclasses.dataclass
class RecurrentWiring:
    """Installed adapters and their instance-local restoration callback."""

    linear_attn_mixers: int = 0
    mamba_modules: tuple[str, ...] = ()
    restore: Callable[[], None] = lambda: None

    def __bool__(self):
        return bool(self.linear_attn_mixers or self.mamba_modules)


def recurrent_plan(models, cp_size):
    from ringmaster.mamba import mamba2_mixers
    from ringmaster.recurrent import kda_mixers, validate_kda, validate_recurrent

    gdn = validate_recurrent(models, cp_size)
    kda = [mixer for model in models for mixer in kda_mixers(model)]
    mamba = [mixer for model in models for mixer in mamba2_mixers(model)]
    validate_kda(kda)
    handled = {id(module) for mixer in gdn + kda + mamba for module in mixer.modules()}
    for model in models:
        for module in model.modules():
            if (
                is_recurrent_mixer(module)
                and id(module) not in handled
                and not any(id(child) in handled for child in module.modules())
            ):
                raise ValueError(
                    f"No context-parallel state adapter for {type(module).__name__}"
                )
    return gdn, kda, mamba


def wire_recurrent_layers(model, *, group=None):
    """Install native recurrent CP adapters; call ``restore`` before reusing the model."""
    import torch.distributed as dist

    from ringmaster.mamba import wire_mamba2
    from ringmaster.recurrent import wire_gated_delta, wire_kda
    from ringmaster.runtime import get_runtime, maybe_runtime

    if group is None:
        runtime = get_runtime()
        if not runtime.config.enabled:
            return RecurrentWiring()
        group = runtime.cp_group
    if dist.get_world_size(group) == 1:
        return RecurrentWiring()
    gdn, kda, mamba = recurrent_plan([model], dist.get_world_size(group))
    runtime = maybe_runtime()
    if (
        (gdn or kda or mamba)
        and runtime is not None
        and runtime.shard_load_balance != "contiguous"
    ):
        raise ValueError("Recurrent context parallelism requires contiguous shards")
    import inspect

    signatures = {
        id(mixer): inspect.signature(mixer.forward) for mixer in gdn + kda + mamba
    }
    restores = []

    def guard(module, args, kwargs):
        runtime = maybe_runtime()
        if runtime is not None and runtime.varlen is not None:
            raise ValueError("Packed recurrent context parallelism is not supported")
        arguments = signatures[id(module)].bind_partial(*args, **kwargs).arguments
        if any(
            arguments.get(key) is not None
            for key in ("cache_params", "past_key_values", "cache")
        ):
            raise ValueError("Recurrent context parallelism requires use_cache=False")

    def restore():
        while restores:
            restores.pop()()

    try:
        for mixer in gdn + kda + mamba:
            handle = mixer.register_forward_pre_hook(guard, with_kwargs=True)
            restores.append(handle.remove)
        for mixers, wire in (
            (gdn, wire_gated_delta),
            (kda, wire_kda),
            (mamba, wire_mamba2),
        ):
            if mixers:
                restores.append(wire(mixers, group))
    except Exception:
        restore()
        raise
    return RecurrentWiring(
        len(gdn) + len(kda), tuple(type(m).__module__ for m in mamba), restore
    )
