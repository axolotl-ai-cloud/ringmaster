"""State-passing CP for recurrent mixers (Mamba/Mamba2, gated linear attention).

A linear recurrence ``h_t = a_t*h_{t-1}+b_t`` is linear in its initial state, so each
rank scans its chunk with ``h0=0``, all-gathers each chunk's (decay ``A_r``, carry
``B_r``) [O(P), seq-independent], prefix-combines previous ranks' (A,B) into the
entering state ``H_in``, and applies the exact correction ``h_t += cumA_t * H_in``.
Generalizes axolotl's Mamba2 CP (PR #3572) to gated/linear attention too.
"""

from __future__ import annotations

import dataclasses

import torch


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
    gathered = [torch.empty_like(stacked) for _ in range(world)]
    dist.all_gather(gathered, stacked, group=group)

    # Prefix-combine previous ranks' (A, B) to get the state entering this chunk.
    h_in = torch.zeros_like(chunk_carry)
    for j in range(rank):
        h_in = gathered[j][0] * h_in + gathered[j][1]

    return local_h + cum_a * h_in.unsqueeze(1)


# Matched case-insensitively against type(module).__name__ so dispatch needs no
# per-architecture imports.
MAMBA_HINTS = ("mamba2mixer", "mambamixer", "mamba2", "mambamixer2")
LINEAR_ATTENTION_HINTS = (
    "gateddeltanet",  # Qwen3-Next / Qwen3.5 gated DeltaNet
    "lineattention",
    "linearattention",
    "gateddelta",
    "deltanet",
)


def is_recurrent_mixer(module) -> bool:
    name = type(module).__name__.lower()
    return any(h in name for h in MAMBA_HINTS + LINEAR_ATTENTION_HINTS)


@dataclasses.dataclass
class RecurrentWiring:
    """Summary of :func:`wire_recurrent_layers`. Truthy iff something was wired."""

    linear_attn_mixers: int = 0
    mamba_modules: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.linear_attn_mixers or self.mamba_modules)


def wire_recurrent_layers(model) -> "RecurrentWiring":
    """Wire CP state-passing into a model's recurrent mixers (single source of truth
    for the axolotl plugin + trl/raw examples; call after ``setup()`` /
    ``set_attn_implementation``, before the first forward).

    Gated linear-attention mixers (Qwen3.5 / Qwen3-Next DeltaNet) get instance-level
    wrapping; Mamba2 SSM mixers (Nemotron-H, Falcon-H1, Granite-MoE-Hybrid) get their
    modeling module's scan kernel wrapped. No-op for pure attention / no mamba-ssm.
    """
    import importlib
    import sys

    from ringmaster.strategies.linear_attn import wrap_linear_attn_instance
    from ringmaster.strategies.mamba import wrap_mamba_scan_for_cp

    text_cfg = getattr(model.config, "get_text_config", lambda: model.config)()
    conv_k = getattr(text_cfg, "linear_conv_kernel_dim", 4)

    ssm_modules, n_linear = set(), 0
    for module in model.modules():
        if not is_recurrent_mixer(module):
            continue
        if hasattr(module, "chunk_gated_delta_rule"):
            wrap_linear_attn_instance(module, conv_k)
            n_linear += 1
        else:
            ssm_modules.add(type(module).__module__)
    mamba_modules = set()
    for mod_name in ssm_modules:
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        if wrap_mamba_scan_for_cp(mod):
            mamba_modules.add(mod_name)
    return RecurrentWiring(n_linear, tuple(sorted(mamba_modules)))
