"""Instance-local adapters for Transformers' Mamba2 scan and convolution hooks."""

import inspect
from types import FunctionType, MethodType

from .recurrent import _rebind_globals


def mamba2_mixers(model):
    from .fla_mamba import fla_mamba_mixers

    mixers = fla_mamba_mixers(model)
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

    def document_ids(x, length):
        from ringmaster.runtime import maybe_runtime
        from ringmaster.ring.varlen_blocks import doc_ids_from_cu as _doc_ids

        runtime = maybe_runtime()
        if runtime is None or runtime.varlen is None:
            return None
        return _doc_ids(runtime.varlen[0], x.device).reshape(
            x.shape[0], length * dist.get_world_size(group)
        )

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
        documents = document_ids(x, x.shape[-1])
        if documents is None:
            output = original_conv(
                torch.cat((prefix, x), dim=-1),
                weight,
                bias,
                activation=activation,
                **kwargs,
            )[..., width:]
        else:
            kwargs.pop("seq_idx", None)
            rows = []
            length = x.shape[-1]
            for row in range(x.shape[0]):
                local_docs = documents[row, rank * length : (rank + 1) * length]
                starts = torch.cat(
                    (
                        local_docs.new_zeros(1),
                        (local_docs[1:] != local_docs[:-1]).nonzero().flatten() + 1,
                        local_docs.new_tensor([length]),
                    )
                ).tolist()
                parts = []
                for start, end in zip(starts[:-1], starts[1:]):
                    values = x[row : row + 1, :, start:end]
                    if start == 0 and rank:
                        prefix_docs = documents[
                            row, rank * length - width : rank * length
                        ]
                        history = prefix[row : row + 1] * (
                            prefix_docs == local_docs[0]
                        ).to(x.dtype)
                        values = torch.cat((history, values), dim=-1)
                    length_before_pad = values.shape[-1]
                    # Hub convolution backward requires aligned strides for singleton slices.
                    values = torch.nn.functional.pad(
                        values, (0, -length_before_pad % 8)
                    ).contiguous()
                    result = original_conv(
                        values, weight, bias, activation=activation, **kwargs
                    )[..., :length_before_pad]
                    parts.append(result[..., -(end - start) :])
                rows.append(torch.cat(parts, dim=-1))
            output = torch.cat(rows, dim=0)
        # Every rank must participate in the halo's backward collective.
        return output + (halos.float().sum() * 0).to(output.dtype)

    def scan(x, dt, A, B, C, chunk_size, **kwargs):
        if kwargs.get("initial_states") is not None:
            raise ValueError("Mamba CP requires uncached sequences")
        if kwargs.get("z") is not None:
            raise ValueError("Mamba CP requires gating outside the scan")
        want_final = kwargs.pop("return_final_states", False)
        documents = document_ids(x, x.shape[1])
        if documents is None:
            output, final = original_scan(
                x, dt, A, B, C, chunk_size, **kwargs, return_final_states=True
            )
        else:
            kwargs.pop("seq_idx", None)
            rows, finals = [], []
            length = x.shape[1]
            for row in range(x.shape[0]):
                local_docs = documents[row, rank * length : (rank + 1) * length]
                starts = torch.cat(
                    (
                        local_docs.new_zeros(1),
                        (local_docs[1:] != local_docs[:-1]).nonzero().flatten() + 1,
                        local_docs.new_tensor([length]),
                    )
                ).tolist()
                parts = []
                for start, end in zip(starts[:-1], starts[1:]):
                    result, state = original_scan(
                        x[row : row + 1, start:end],
                        dt[row : row + 1, start:end],
                        A,
                        B[row : row + 1, start:end],
                        C[row : row + 1, start:end],
                        chunk_size,
                        **kwargs,
                        return_final_states=True,
                    )
                    parts.append(result)
                rows.append(torch.cat(parts, dim=1))
                finals.append(state)
            output, final = torch.cat(rows), torch.cat(finals)
        effective_dt = dt.float()
        if kwargs.get("dt_bias") is not None:
            effective_dt = effective_dt + kwargs["dt_bias"].float()
        if kwargs.get("dt_softplus"):
            effective_dt = torch.nn.functional.softplus(effective_dt)
        effective_dt = effective_dt.clamp(*kwargs.get("dt_limit", (0.0, float("inf"))))
        cumulative = (effective_dt * A.float()).cumsum(1)
        decay = cumulative[:, -1].exp()
        continuation = None
        if documents is not None:
            length = x.shape[1]
            local_docs = documents[:, rank * length : (rank + 1) * length]
            previous = documents[:, rank * length - 1] if rank else local_docs[:, 0] - 1
            continuation = local_docs == previous[:, None]
            decay = decay * continuation.all(dim=1, keepdim=True)
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
        if continuation is not None:
            correction = correction * continuation[:, :, None, None]
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
    fla_restore = None

    def restore():
        if fla_restore is not None:
            fla_restore()
        for module, existed, forward in reversed(originals):
            if existed:
                module.forward = forward
            else:
                del module.forward

    try:
        from .fla_mamba import fla_mamba_mixers, wire_fla_mamba

        fla = [mixer for mixer in mixers if fla_mamba_mixers(mixer)]
        fla_restore = wire_fla_mamba(fla, group) if fla else None
        for mixer in mixers:
            if mixer in fla:
                continue
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
