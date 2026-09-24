"""Instance-local adapters for Transformers' Mamba2 scan and convolution hooks."""

import inspect
from types import FunctionType, MethodType

from .recurrent import _rebind_globals


def mamba2_mixers(model):
    mixers = []
    for module in model.modules():
        forward = getattr(module.forward, "__func__", None)
        raw = inspect.unwrap(forward) if forward is not None else None
        if isinstance(raw, FunctionType) and {
            "mamba2_chunk_scan",
            "causal_conv1d_fn",
            "mamba2_split_conv1d_scan_combined",
        } <= set(raw.__code__.co_names):
            mixers.append(module)
    return mixers


def _bindings(raw, group):
    import torch
    import torch.distributed as dist

    from ringmaster.cp_collectives import all_gather_cp

    original_scan = raw.__globals__["mamba2_chunk_scan"]
    original_conv = raw.__globals__["causal_conv1d_fn"]
    rank = dist.get_rank(group)

    def conv(x, weight, bias=None, activation=None, **kwargs):
        width = weight.shape[-1] - 1
        if width == 0:
            return original_conv(x, weight, bias, activation=activation, **kwargs)
        if x.shape[-1] < width:
            raise ValueError(
                "Mamba CP shards must be at least convolution width - 1 tokens"
            )
        halos = all_gather_cp(x[..., -width:].contiguous(), group)
        prefix = halos[rank - 1] if rank else torch.zeros_like(halos[0])
        output = original_conv(
            torch.cat((prefix, x), dim=-1),
            weight,
            bias,
            activation=activation,
            **kwargs,
        )[..., width:]
        # Every rank must participate in the halo's backward collective.
        return output + (halos.float().sum() * 0).to(output.dtype)

    def scan(x, dt, A, B, C, chunk_size, **kwargs):
        if (
            kwargs.get("initial_states") is not None
            or kwargs.get("seq_idx") is not None
        ):
            raise ValueError("Mamba CP requires uncached, unpacked sequences")
        if kwargs.get("z") is not None:
            raise ValueError("Mamba CP requires gating outside the scan")
        want_final = kwargs.pop("return_final_states", False)
        output, final = original_scan(
            x, dt, A, B, C, chunk_size, **kwargs, return_final_states=True
        )
        effective_dt = dt.float()
        if kwargs.get("dt_bias") is not None:
            effective_dt = effective_dt + kwargs["dt_bias"].float()
        if kwargs.get("dt_softplus"):
            effective_dt = torch.nn.functional.softplus(effective_dt)
        effective_dt = effective_dt.clamp(*kwargs.get("dt_limit", (0.0, float("inf"))))
        cumulative = (effective_dt * A.float()).cumsum(1)
        decay = cumulative[:, -1].exp()
        batch, heads = decay.shape
        packed = torch.cat((decay, final.float().reshape(batch, -1)), dim=1)
        states = all_gather_cp(packed, group)
        entering = torch.zeros_like(final, dtype=torch.float32)
        for predecessor in range(rank):
            entering = states[predecessor, :, :heads, None, None] * entering + states[
                predecessor, :, heads:
            ].reshape_as(entering)
        expanded_c = C.float().repeat_interleave(heads // C.shape[2], dim=2)
        correction = torch.einsum("bthn,bhdn->bthd", expanded_c, entering)
        correction = correction * cumulative.exp().unsqueeze(-1)
        output = (
            output + correction.to(output.dtype) + (states.sum() * 0).to(output.dtype)
        )
        final = final + (decay[..., None, None] * entering).to(final.dtype)
        return (output, final) if want_final else output

    return {
        "mamba2_chunk_scan": scan,
        "causal_conv1d_fn": conv,
        "mamba2_split_conv1d_scan_combined": lambda *args, **kwargs: None,
    }


def _uses_fused_norm(function):
    function = getattr(function, "forward", function)
    function = getattr(function, "__func__", function)
    if not isinstance(function, FunctionType):
        raise ValueError(
            "Cannot determine the selected Mamba fused kernel's normalization"
        )
    if (
        function.__name__ == "mamba_split_conv1d_scan_combined"
        and function.__module__.endswith(".ssd_combined")
    ):
        return True
    closure = inspect.getclosurevars(function).nonlocals
    for name in ("implementation", "func"):
        if name in closure:
            return _uses_fused_norm(closure[name])
    for name in function.__code__.co_names:
        if name.endswith("mamba_split_conv1d_scan_combined"):
            implementation = function.__globals__.get(name)
            if implementation is not None and implementation is not function:
                return _uses_fused_norm(implementation)
    if (
        function.__module__.startswith("transformers.models.")
        and function.__name__ == "mamba2_split_conv1d_scan_combined"
        and not function.__code__.co_names
    ):
        return False
    raise ValueError("Cannot determine the selected Mamba fused kernel's normalization")


def _norm_forward(mixer, fused):
    import torch

    original = mixer.norm.forward
    uses_fused = _uses_fused_norm(fused)

    def forward(norm, hidden_states, gate=None):
        if not mixer.training or not uses_fused:
            return original(hidden_states, gate)
        # Fused Mamba normalizes each SSM group; Transformers' unfused path does not.
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(gate.float())
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(
            *shape[:-1], mixer.n_groups, shape[-1] // mixer.n_groups
        )
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + norm.variance_epsilon)
        return norm.weight * hidden_states.reshape(shape).to(dtype)

    return MethodType(forward, mixer.norm)


def wire_mamba2(mixers, group):
    originals = []

    def restore():
        for module, existed, forward in reversed(originals):
            if existed:
                module.forward = forward
            else:
                del module.forward

    try:
        for mixer in mixers:
            forward = mixer.forward.__func__
            raw = inspect.unwrap(forward)
            if getattr(mixer, "mamba_rms_norm", True) and (
                mixer.n_groups > 1 or getattr(mixer.norm, "norm_before_gate", False)
            ):
                fused = raw.__globals__["mamba2_split_conv1d_scan_combined"]
                if not hasattr(mixer.norm, "variance_epsilon"):
                    raise ValueError("Mamba CP requires a supported gated RMSNorm")
                norm_forward = _norm_forward(mixer, fused)
                originals.append(
                    (
                        mixer.norm,
                        "forward" in vars(mixer.norm),
                        vars(mixer.norm).get("forward"),
                    )
                )
                mixer.norm.forward = norm_forward
            adapted_forward = MethodType(
                _rebind_globals(forward, _bindings(raw, group)), mixer
            )
            originals.append(
                (mixer, "forward" in vars(mixer), vars(mixer).get("forward"))
            )
            mixer.forward = adapted_forward
    except Exception:
        restore()
        raise
    return restore
