"""Gated linear-attention (DeltaNet) + causal-conv CP — owned by ringmaster.

Qwen3.5 / Qwen3-Next linear-attention layers use flash-linear-attention's gated
delta rule + a causal conv1d. Two CP paths:

* **native** (preferred): fla (>=0.5.1) ``build_cp_context`` + ``cp_context`` — the
  parallel "True CP" path. Used when fla exposes ``cp_context`` AND its backward
  works in the current toolchain. On cu13/sm_120 Blackwell, fla's gated-delta
  backward dispatches to a TileLang kernel whose warp-specialized MMA pass crashes
  (CUDA_ERROR_MISALIGNED_ADDRESS); :func:`_apply_fla_sm120_shim` surgically disables
  just that pass so the native path stays usable.
* **torch fallback**: opt-in via RINGMASTER_NO_FLA_SHIM=1 (or any toolchain where
  the native backward can't run) — routes the mixer through its own ``torch_*``
  reference kernels (pure autograd) wrapped with an autograd-aware CP state pass +
  conv halo.

``wrap_linear_attn_instance`` patches a mixer instance, picking the path that can
both forward AND backward in the current environment.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.distributed as dist

from ringmaster.strategies.mamba import _cp_group, is_cp_active


def _global_cu_seqlens(local_len: int, device, group) -> torch.Tensor:
    world = dist.get_world_size(group)
    return torch.tensor([0, local_len * world], device=device, dtype=torch.long)


# fla TileLang kernel modules whose JITImpl ``pass_configs`` we patch on sm_120.
_FLA_TILELANG_MODULES = (
    "fla.ops.common.backends.tilelang.chunk_bwd",
    "fla.ops.common.backends.tilelang.parallel_attn_bwd",
    "fla.ops.common.backends.tilelang.parallel_attn_fwd",
    "fla.ops.kda.backends.tilelang.chunk_bwd_dqkg",
)


@lru_cache(maxsize=1)
def _apply_fla_sm120_shim() -> bool:
    """Disable TileLang warp specialization on fla's TileLang kernels for sm_120.

    fla's gated-delta backward (``chunk_bwd_dqkwg_tilelang``) crashes on sm_120: TileLang
    0.1.9's warp-specialized MMA emits misaligned shared-mem descriptors
    (CUDA_ERROR_MISALIGNED_ADDRESS, fla #913). Rather than the sledgehammer
    ``FLA_DISABLE_BACKEND_DISPATCH=1`` (forces every op onto Triton), inject
    ``TL_DISABLE_WARP_SPECIALIZED`` into just these kernels' ``pass_configs`` —
    bit-for-bit identical (warp-spec is only a pipelining transform). TileLang compiles
    lazily per-shape, so mutating before the first backward works. Returns True if in
    effect (or unnecessary on cuda<13); opt out with RINGMASTER_NO_FLA_SHIM=1.
    """
    import importlib
    import os

    if os.environ.get("RINGMASTER_NO_FLA_SHIM") == "1":
        return False
    cuda = int((torch.version.cuda or "0").split(".")[0])
    if cuda < 13:
        return True

    try:
        import tilelang
    except ImportError:
        return False
    key = tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED

    patched = 0
    for name in _FLA_TILELANG_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for obj in vars(mod).values():
            pc = getattr(obj, "pass_configs", None)
            if isinstance(pc, dict):  # a tilelang JITImpl
                pc.setdefault(key, True)
                patched += 1
    return patched > 0


@lru_cache(maxsize=1)
def _fla_backward_ok() -> bool:
    """Whether fla's gated-delta backward runs here: True on cuda<13, else the sm_120
    warp-spec shim (:func:`_apply_fla_sm120_shim`). Force RINGMASTER_FORCE_FLA_BWD=1;
    torch fallback RINGMASTER_NO_FLA_SHIM=1."""
    import os

    if os.environ.get("RINGMASTER_FORCE_FLA_BWD") == "1":
        return True
    cuda = int((torch.version.cuda or "0").split(".")[0])
    if cuda < 13:
        return True
    return _apply_fla_sm120_shim()


@lru_cache(maxsize=1)
def _fla_has_cp_context() -> bool:
    try:
        import inspect

        from fla.ops.cp import build_cp_context  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return "cp_context" in inspect.signature(chunk_gated_delta_rule).parameters
    except Exception:
        return False


from ringmaster.cp_collectives import AllGatherCP as _AllGather


def _native_cp(mixer, conv_kernel_size, build_cp_context):
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

    for fn_name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        kernel = getattr(mixer, fn_name, None)
        if kernel is None:
            continue

        def make(kernel):
            def wrapped(*args, **kw):
                if not is_cp_active() or kw.get("initial_state") is not None or kw.get("cp_context") is not None:
                    return kernel(*args, **kw)
                group = _cp_group()
                q = kw.get("q", args[0] if args else None)
                ctx = build_cp_context(_global_cu_seqlens(q.shape[1], q.device, group), group=group)
                kw = {k: v for k, v in kw.items() if k not in ("cu_seqlens", "initial_state", "output_final_state")}
                kw["cp_context"] = ctx
                return kernel(*args, **kw)
            return wrapped

        setattr(mixer, fn_name, make(kernel))

    orig_conv = mixer.causal_conv1d_fn
    mixer._orig_causal_conv1d_fn = orig_conv

    def conv_cp(*args, **kw):
        if not is_cp_active():
            return orig_conv(*args, **kw)
        x = kw.get("x", args[0] if args else None)
        group = _cp_group()
        ctx = build_cp_context(_global_cu_seqlens(x.shape[-1], x.device, group), group=group,
                               conv1d_kernel_size=conv_kernel_size)
        y, _ = fla_causal_conv1d(x.transpose(1, 2).contiguous(), weight=kw.get("weight"),
                                 bias=kw.get("bias"), activation=kw.get("activation"), cp_context=ctx)
        return y.transpose(1, 2)

    mixer.causal_conv1d_fn = conv_cp


def _torch_fallback_cp(mixer, conv_kernel_size):
    """Autograd-aware CP using the mixer's torch reference kernels (no fla Triton).
    Exact for 2 ranks (all-gather of finals/tails); >2 needs the native path."""
    import sys

    mod = sys.modules[type(mixer).__module__]
    torch_gdn = getattr(mod, "torch_chunk_gated_delta_rule", None)

    if torch_gdn is not None and getattr(mixer, "chunk_gated_delta_rule", None) is not None:
        def gdn_cp(*args, **kw):
            if not is_cp_active() or kw.get("initial_state") is not None:
                return torch_gdn(*args, **kw)
            group = _cp_group()
            world = dist.get_world_size(group)
            rank = dist.get_rank(group)
            kw_local = {**kw, "initial_state": None, "output_final_state": True}
            out, fin = torch_gdn(*args, **kw_local)
            finals = _AllGather.apply(fin, group)  # [world, ...], autograd-aware
            if rank > 0:
                kw_real = {**kw, "initial_state": finals[rank - 1], "output_final_state": True}
                out, fin = torch_gdn(*args, **kw_real)
            else:
                # rank 0 doesn't consume the gather; keep it in the graph so its
                # backward all-reduce still fires (else ranks deadlock).
                out = out + 0.0 * finals.float().sum().to(out.dtype)
            return out, (fin if kw.get("output_final_state") else None)

        mixer.chunk_gated_delta_rule = gdn_cp

    # Conv: package conv (CUDA ext, backward works) + autograd-aware halo for 2 ranks.
    orig_conv = mixer.causal_conv1d_fn
    if orig_conv is not None:
        pad = conv_kernel_size - 1

        def conv_cp(*args, **kw):
            if not is_cp_active() or pad <= 0:
                return orig_conv(*args, **kw)
            x = kw.get("x", args[0] if args else None)  # [B, C, L]
            group = _cp_group()
            rank = dist.get_rank(group)
            tails = _AllGather.apply(x[..., -pad:], group)  # [world, B, C, pad]
            if rank > 0:
                halo = tails[rank - 1]
            else:
                halo = torch.zeros_like(x[..., -pad:])
            x_aug = torch.cat([halo, x], dim=-1)
            kw2 = {**kw, "x": x_aug} if "x" in kw else kw
            out = orig_conv(**kw2) if "x" in kw else orig_conv(x_aug, *args[1:], **kw)
            out = out[..., -x.shape[-1]:]
            if rank == 0:
                # keep the gather in rank 0's graph (symmetric backward all-reduce)
                out = out + 0.0 * tails.float().sum().to(out.dtype)
            return out

        mixer.causal_conv1d_fn = conv_cp


def wrap_linear_attn_instance(mixer, conv_kernel_size: int):
    if getattr(mixer, "_cp_instance_wrapped", False):
        return
    mixer._cp_instance_wrapped = True

    if _fla_has_cp_context() and _fla_backward_ok():
        from fla.ops.cp import build_cp_context

        _native_cp(mixer, conv_kernel_size, build_cp_context)
    else:
        _torch_fallback_cp(mixer, conv_kernel_size)
