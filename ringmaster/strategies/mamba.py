"""Mamba2 SSM context-parallelism — owned by ringmaster.

Ported from axolotl so the per-architecture SSM-kernel wiring lives in one place.
An SSM output is linear in its initial state, so under CP each rank runs its local
scan, receives the previous rank's final state via one P2P hop, and applies an
exact additive correction (Tri Dao's Mamba-2 systems approach) — no ring attention
for SSM layers. Used by hybrid architectures (Nemotron-H, Falcon-H1,
Granite-MoE-Hybrid).

This is the concrete realization of the linear-recurrence primitive in
``state_passing.py`` for the mamba-ssm chunk-scan kernel.
"""

from __future__ import annotations

import functools

import torch
import torch.distributed as dist

from ringmaster.runtime import maybe_runtime


def _cp_group():
    rt = maybe_runtime()
    if rt is not None and rt.cp_group is not None:
        # SSM state passes along the sequence-shard order — the full CP group.
        return rt.cp_group
    # Fall back to a host framework's ring group (e.g. axolotl's legacy CP path) so
    # the same correction works whether driven by ringmaster.setup() or a host.
    try:
        from axolotl.monkeypatch.ring_attn import get_ring_attn_group

        return get_ring_attn_group()
    except (ImportError, RuntimeError):
        return None


def is_cp_active() -> bool:
    group = _cp_group()
    return group is not None and dist.get_world_size(group) > 1


def ring_shift_ssm_state(h_final: torch.Tensor) -> torch.Tensor:
    """P2P ring: send h_final to rank+1, receive from rank-1 within the CP group.
    Rank 0 receives zeros (no predecessor)."""
    group = _cp_group()
    h_prev = torch.zeros_like(h_final)
    if group is None or dist.get_world_size(group) <= 1:
        return h_prev

    local_rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    ranks = dist.get_process_group_ranks(group)
    prev_global = ranks[(local_rank - 1) % world_size]
    next_global = ranks[(local_rank + 1) % world_size]

    send_op = dist.P2POp(dist.isend, h_final.contiguous(), next_global, group=group)
    recv_op = dist.P2POp(dist.irecv, h_prev, prev_global, group=group)
    for req in dist.batch_isend_irecv([send_op, recv_op]):
        req.wait()

    if local_rank == 0:
        h_prev.zero_()
    return h_prev


def cp_state_prefix(ssm_state, chunk_decay, group, *, single_hop=False):
    """State entering this rank's chunk: all-gather each rank's chunk decay ``A_r``
    ([b, nheads]) and local final ``B_r`` (=``ssm_state``, h0=0), then LOCALLY prefix-
    combine the predecessors — ``H_in_r = sum_{j<r} (prod_{j<k<r} A_k) B_j``. One
    autograd-aware all-gather (decay+state packed together) + an O(P) local loop; no
    serial P2P. Exact for any P. ``single_hop=True`` keeps only the immediate
    predecessor (packed sequences, where cross-doc resets make the prefix doc-dependent).

    Returns ``(h_prev, zero_keep)``; add ``zero_keep`` (None except on rank 0) to the
    output so the gather stays in rank 0's graph and its backward all-reduce fires."""
    from ringmaster.cp_collectives import all_gather_cp

    rank = dist.get_rank(group)
    b, nh = ssm_state.shape[0], chunk_decay.shape[1]
    packed = torch.cat(
        [chunk_decay.reshape(b, -1).to(ssm_state.dtype), ssm_state.reshape(b, -1)], dim=1
    )
    gathered = all_gather_cp(packed, group)  # [world, b, nheads + state]
    zero_keep = 0.0 * gathered.float().sum() if rank == 0 else None

    def _final(j):
        return gathered[j][:, nh:].reshape_as(ssm_state)

    if rank == 0:
        return torch.zeros_like(ssm_state), zero_keep
    if single_hop:
        return _final(rank - 1), zero_keep
    h_prev = torch.zeros_like(ssm_state)
    for j in range(rank):
        h_prev = gathered[j][:, :nh][..., None, None] * h_prev + _final(j)
    return h_prev, zero_keep


def mamba2_cp_correction(
    out: torch.Tensor,
    h_final: torch.Tensor,
    C: torch.Tensor,
    cum_A: torch.Tensor,
    h_prev: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the analytic contribution of h_prev (received from rank-1) to the SSM
    output and final state. seq_idx masks the correction to zero for packed
    sequences that start fresh on this rank."""
    if not h_prev.any():
        return out, h_final

    B, T, _ = out.shape
    n_groups = C.shape[2]
    heads_per_group = num_heads // n_groups

    decay = torch.exp(cum_A).float()  # [B, T, H]
    prop_state = decay[:, :, :, None, None] * h_prev[:, None, :, :, :].float()
    C_expanded = C.float().repeat_interleave(heads_per_group, dim=2)  # [B, T, H, n]
    delta_y = torch.einsum("bthn,bthdn->bthd", C_expanded, prop_state)

    if seq_idx is not None:
        mask = (seq_idx == 0).to(delta_y.dtype).unsqueeze(-1).unsqueeze(-1)
        delta_y = delta_y * mask

    delta_y = delta_y.reshape(B, T, num_heads * head_dim).to(out.dtype)
    corrected_out = out + delta_y

    if seq_idx is not None and seq_idx[:, -1].any():
        corrected_h_final = h_final
    else:
        decay_final = decay[:, -1, :, None, None]
        corrected_h_final = h_final + (decay_final * h_prev.float()).to(h_final.dtype)

    return corrected_out, corrected_h_final


def prefer_local_mamba_kernels() -> bool:
    """Route transformers' kernel loader to the pip mamba-ssm / causal-conv1d.

    The kernels-hub ``mamba-ssm`` prebuilt targets the OLD causal_conv1d ABI (raw
    ``causal_conv1d_cuda.causal_conv1d_fwd``), which breaks against pip causal_conv1d
    >=1.6. Preseating ``_KERNEL_MODULE_MAPPING`` with the importable pip modules makes
    ``lazy_load_kernel`` short-circuit to them. Run before the first mamba forward;
    idempotent. Returns True if any kernel was preset.
    """
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return False
    import importlib
    from types import ModuleType

    mapping = hub_kernels._KERNEL_MODULE_MAPPING
    presets = {
        "mamba-ssm": "mamba_ssm",
        "falcon_mamba-ssm": "mamba_ssm",
        "causal-conv1d": "causal_conv1d",
    }
    patched = 0
    for key, modname in presets.items():
        if isinstance(mapping.get(key), ModuleType):
            continue
        try:
            mapping[key] = importlib.import_module(modname)
            patched += 1
        except ImportError:
            continue
    return patched > 0


def ensure_causal_conv1d_cuda_export() -> bool:
    """Re-attach ``causal_conv1d_cuda`` for the kernels-hub mamba build.

    It imports ``from causal_conv1d.causal_conv1d_interface import causal_conv1d_cuda``
    (old layout), but pip causal_conv1d >=1.6 moved it and dropped that re-export → the
    build silently sets it None and the fused path asserts. The compiled ext is still
    importable top-level, so re-attach it. Run before first mamba forward; idempotent.
    """
    try:
        import causal_conv1d.causal_conv1d_interface as cci
    except Exception:
        return False
    if getattr(cci, "causal_conv1d_cuda", None) is not None:
        return True
    try:
        import causal_conv1d_cuda  # compiled ext, top-level importable
    except Exception:
        return False
    cci.causal_conv1d_cuda = causal_conv1d_cuda
    return True


def ensure_mamba_kernels_loaded(target_module):
    """Eagerly resolve mamba-ssm / causal-conv1d globals on target_module (they are
    lazily loaded inside Mixer.__init__ on transformers >= 5.5)."""
    if getattr(target_module, "mamba_chunk_scan_combined", None) is not None:
        return
    prefer_local_mamba_kernels()
    ensure_causal_conv1d_cuda_export()
    try:
        from transformers.integrations.hub_kernels import lazy_load_kernel
        from transformers.utils.import_utils import resolve_internal_import
    except ImportError:
        return

    causal_conv1d = lazy_load_kernel("causal-conv1d")
    if causal_conv1d is not None:
        target_module.causal_conv1d_update = getattr(causal_conv1d, "causal_conv1d_update", None)
        target_module.causal_conv1d_fn = getattr(causal_conv1d, "causal_conv1d_fn", None)

    mamba_ssm = lazy_load_kernel("mamba-ssm")
    if mamba_ssm is not None:
        target_module.selective_state_update = resolve_internal_import(
            mamba_ssm, chained_path="ops.triton.selective_state_update.selective_state_update"
        )
        target_module.mamba_chunk_scan_combined = resolve_internal_import(
            mamba_ssm, chained_path="ops.triton.ssd_combined.mamba_chunk_scan_combined"
        )
        target_module.mamba_split_conv1d_scan_combined = resolve_internal_import(
            mamba_ssm, chained_path="ops.triton.ssd_combined.mamba_split_conv1d_scan_combined"
        )

    target_module.is_fast_path_available = all(
        (
            getattr(target_module, "selective_state_update", None),
            getattr(target_module, "mamba_chunk_scan_combined", None),
            getattr(target_module, "mamba_split_conv1d_scan_combined", None),
            getattr(target_module, "causal_conv1d_fn", None),
            getattr(target_module, "causal_conv1d_update", None),
        )
    )


def wrap_mamba_scan_for_cp(target_module) -> bool:
    """Wrap mamba_chunk_scan_combined in target_module to apply the CP correction.

    Returns True if the module actually has a Mamba2 chunk-scan that was (or is
    already) wrapped, False for modules with no SSM scan (e.g. pure gated
    linear-attention models like Qwen3.5 — no-op).
    """
    if getattr(target_module, "_cp_scan_wrapped", False):
        return True
    ensure_mamba_kernels_loaded(target_module)
    if getattr(target_module, "mamba_chunk_scan_combined", None) is None:
        return False

    original_scan = target_module.mamba_chunk_scan_combined

    @functools.wraps(original_scan)
    def _cp_scan_wrapper(*args, **kwargs):
        cp_active = is_cp_active()
        if cp_active:
            kwargs["return_final_states"] = True
        result = original_scan(*args, **kwargs)
        if not cp_active:
            return result

        scan_output, ssm_state = result
        if ssm_state is None:
            return result

        group = _cp_group()

        dt_arg = kwargs.get("dt", args[1] if len(args) > 1 else None)
        A_arg = kwargs.get("A", args[2] if len(args) > 2 else None)
        C_arg = kwargs.get("C", args[4] if len(args) > 4 else None)
        if dt_arg is None or A_arg is None or C_arg is None:
            raise ValueError("wrap_mamba_scan_for_cp requires dt, A, C (positional or kwargs).")
        dt_bias = kwargs.get("dt_bias")
        dt_softplus = kwargs.get("dt_softplus", False)
        seq_idx = kwargs.get("seq_idx")
        dt_eff = (
            torch.nn.functional.softplus(dt_arg + (dt_bias if dt_bias is not None else 0))
            if dt_softplus else dt_arg
        )
        cum_A = torch.cumsum(A_arg[None, None, :] * dt_eff, dim=1)  # [b, T, nheads]

        # exact cross-rank entering state (single-hop for packed; see cp_state_prefix)
        nh = A_arg.shape[0]
        chunk_decay = torch.exp(cum_A[:, -1])  # A_r per head: [b, nheads]
        h_prev, zero_keep = cp_state_prefix(
            ssm_state, chunk_decay, group, single_hop=seq_idx is not None
        )

        x = args[0]
        head_dim = x.shape[3] if x.ndim == 4 else x.shape[2] // nh
        B_dim, T_dim = x.shape[0], x.shape[1]

        scan_flat = scan_output.view(B_dim, T_dim, -1)
        scan_flat, ssm_state = mamba2_cp_correction(
            scan_flat, ssm_state, C_arg, cum_A, h_prev,
            num_heads=nh, head_dim=head_dim, seq_idx=seq_idx,
        )
        if zero_keep is not None:
            scan_flat = scan_flat + zero_keep.to(scan_flat.dtype)
        scan_output = scan_flat.view(scan_output.shape)
        return scan_output, ssm_state

    target_module.mamba_chunk_scan_combined = _cp_scan_wrapper
    target_module._cp_scan_wrapped = True
    return True
