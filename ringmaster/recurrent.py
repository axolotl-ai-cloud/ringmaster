"""Instance-local native FLA CP adapters for GDN and KDA mixers."""

import inspect
from functools import update_wrapper
from types import FunctionType, MethodType


def is_recurrent_mixer(module):
    name = type(module).__name__.lower()
    return any(
        hint in name
        for hint in (
            "mambamixer",
            "mamba2mixer",
            "mambamixer2",
            "gateddelta",
            "deltanet",
            "linearattention",
            "lineattention",
            "shortconv",
            "deltaattention",
            "lightningattention",
        )
    )


def gated_delta_mixers(model):
    return [
        module
        for module in model.modules()
        if "gateddelta" in type(module).__name__.lower()
        or hasattr(module, "chunk_gated_delta_rule")
    ]


def require_fla_cp():
    """Require importable native CP kernels, rather than package metadata alone."""
    try:
        from fla.modules.convolution import causal_conv1d
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise ValueError(
            "Gated-delta context parallelism requires importable FLA CP kernels; "
            "install axolotl-ringmaster[fla]. CP > 2 cannot use the torch state-passing fallback."
        ) from exc
    if not all(
        "cp_context" in inspect.signature(kernel).parameters
        for kernel in (chunk_gated_delta_rule, causal_conv1d)
    ):
        raise ValueError(
            "Installed FLA lacks native cp_context support; install axolotl-ringmaster[fla]"
        )
    return build_cp_context, chunk_gated_delta_rule, causal_conv1d


def validate_recurrent(models, cp_size):
    """Validate before attention or recurrent functions are changed."""
    mixers = [m for model in models for m in gated_delta_mixers(model)]
    if mixers:
        require_fla_cp()
        for mixer in mixers:
            if hasattr(mixer, "chunk_gated_delta_rule"):
                continue
            forward = getattr(mixer.forward, "__func__", None)
            raw = inspect.unwrap(forward) if forward is not None else None
            if not isinstance(raw, FunctionType) or not {
                "torch_chunk_gated_delta_rule",
                "causal_conv1d_fn",
            } <= set(raw.__code__.co_names):
                raise ValueError(
                    f"Unsupported gated-delta forward: {type(mixer).__name__}"
                )
        if cp_size < 2:
            raise ValueError("Gated-delta CP requires at least two ranks")
    return mixers


def _rebind_globals(function, replacements):
    """Retain decorated-forward closures while replacing only their kernel bindings."""
    wrapped = getattr(function, "__wrapped__", None)
    rebound = _rebind_globals(wrapped, replacements) if wrapped is not None else None
    closure = function.__closure__
    if closure and wrapped is not None:
        closure = tuple(
            (lambda value: lambda: value)(rebound).__closure__[0]
            if cell.cell_contents is wrapped
            else cell
            for cell in closure
        )
    clone = FunctionType(
        function.__code__,
        function.__globals__ | replacements,
        function.__name__,
        function.__defaults__,
        closure,
    )
    update_wrapper(clone, function)
    clone.__kwdefaults__ = function.__kwdefaults__
    if wrapped is not None:
        clone.__wrapped__ = rebound
    return clone


def wire_gated_delta(mixers, cp_group):
    """Install native FLA state passing and convolution halos; return an undo callback."""
    import torch
    import torch.distributed as dist

    build_context, chunk_gdn, causal_conv = require_fla_cp()
    world = dist.get_world_size(cp_group)
    originals = []

    def context(x, *, conv_size=None):
        if x.shape[0] != 1:
            raise ValueError(
                "FLA context parallelism currently requires micro_batch_size: 1"
            )
        if x.shape[1] == 0 or (conv_size is not None and x.shape[1] < conv_size - 1):
            raise ValueError(
                "FLA CP shards must contain at least convolution width - 1 tokens"
            )
        cu = torch.tensor([0, x.shape[1] * world], dtype=torch.long, device=x.device)
        return build_context(cu, group=cp_group, conv1d_kernel_size=conv_size)

    def delta(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        if initial_state is not None or output_final_state:
            raise ValueError(
                "Gated-delta CP requires use_cache=False and no initial_state"
            )
        cu = kwargs.pop("cu_seqlens", None)
        if cu is not None and (
            cu.numel() != 2 or cu[0].item() != 0 or cu[-1].item() != q.shape[1]
        ):
            raise ValueError("GDN CP currently requires unpacked contiguous shards")
        kwargs.pop("cu_seqlens_cpu", None)
        return chunk_gdn(
            q,
            k,
            v,
            g,
            beta,
            cp_context=context(q),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

    def conv(x, weight, bias=None, activation=None, **kwargs):
        x = x.transpose(1, 2).contiguous()
        result, _ = causal_conv(
            x,
            weight=weight,
            bias=bias,
            activation=activation,
            cp_context=context(x, conv_size=weight.shape[-1]),
        )
        return result.transpose(1, 2)

    def replace(obj, name, value):
        originals.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, value)

    for mixer in mixers:
        if hasattr(mixer, "chunk_gated_delta_rule"):
            replace(mixer, "chunk_gated_delta_rule", delta)
            replace(mixer, "causal_conv1d_fn", conv)
        else:
            forward = mixer.forward.__func__
            raw = inspect.unwrap(forward)
            if not {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"} <= set(
                raw.__code__.co_names
            ):
                raise ValueError(
                    f"Unsupported gated-delta forward: {type(mixer).__name__}"
                )
            replace(
                mixer,
                "forward",
                MethodType(
                    _rebind_globals(
                        forward,
                        {
                            "torch_chunk_gated_delta_rule": delta,
                            "torch_recurrent_gated_delta_rule": delta,
                            "causal_conv1d_fn": conv,
                        },
                    ),
                    mixer,
                ),
            )

    def restore():
        for obj, name, existed, value in reversed(originals):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)

    return restore


def kda_mixers(model):
    mixers = []
    for module in model.modules():
        forward = getattr(module.forward, "__func__", None)
        raw = inspect.unwrap(forward) if forward is not None else None
        if isinstance(raw, FunctionType) and "chunk_kda" in raw.__code__.co_names:
            mixers.append(module)
    return mixers


def require_kda_cp():
    build_context, _, causal_conv = require_fla_cp()
    from fla.ops.kda import chunk_kda

    if "cp_context" not in inspect.signature(chunk_kda).parameters:
        raise ValueError("Installed FLA lacks native KDA context parallelism")
    return build_context, chunk_kda, causal_conv


def validate_kda(mixers):
    if not mixers:
        return
    require_kda_cp()
    from fla.modules import ShortConvolution

    for mixer in mixers:
        if not all(
            isinstance(getattr(mixer, name, None), ShortConvolution)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        ):
            raise ValueError(f"Unsupported KDA convolutions: {type(mixer).__name__}")


def wire_kda(mixers, cp_group):
    """Bind native FLA KDA and convolution CP kernels without changing globals."""
    from functools import wraps

    import torch
    import torch.distributed as dist

    validate_kda(mixers)
    build_context, chunk_kda, causal_conv = require_kda_cp()
    world = dist.get_world_size(cp_group)
    originals = []

    def context(x, cu_seqlens=None, conv_size=None):
        if x.shape[0] != 1:
            raise ValueError("KDA CP requires a batch size of one")
        if cu_seqlens is not None and (
            cu_seqlens.numel() != 2
            or cu_seqlens[0].item() != 0
            or cu_seqlens[-1].item() != x.shape[1]
        ):
            raise ValueError("KDA CP currently requires unpacked contiguous shards")
        if x.shape[1] == 0 or (conv_size is not None and x.shape[1] < conv_size - 1):
            raise ValueError(
                "FLA CP shards must contain at least convolution width - 1 tokens"
            )
        cu = torch.tensor([0, x.shape[1] * world], dtype=torch.long, device=x.device)
        return build_context(cu, group=cp_group, conv1d_kernel_size=conv_size)

    def delta(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        **kwargs,
    ):
        if initial_state is not None or output_final_state:
            raise ValueError("KDA CP does not support cached states")
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cp_context=context(q, cu_seqlens),
            **kwargs,
        )

    def conv(
        self,
        x,
        residual=None,
        mask=None,
        cache=None,
        output_final_state=False,
        cu_seqlens=None,
        **kwargs,
    ):
        if (
            cache is not None
            or output_final_state
            or mask is not None
            or residual is not None
        ):
            raise ValueError("KDA CP requires uncached, unmasked short convolutions")
        return causal_conv(
            x,
            self.weight.squeeze(1),
            self.bias,
            activation=self.activation,
            cp_context=context(x, cu_seqlens, self.kernel_size[0]),
        )

    def guard(forward):
        signature = inspect.signature(forward)

        @wraps(forward)
        def checked(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            if any(
                arguments.get(name) is not None
                for name in ("cache_params", "past_key_values")
            ):
                raise ValueError("KDA CP requires use_cache=False")
            mask = arguments.get("attention_mask")
            if mask is not None and (mask.ndim != 2 or not torch.all(mask == 1)):
                raise ValueError("KDA CP currently requires dense, unpacked sequences")
            return forward(*args, **kwargs)

        return checked

    def replace(obj, name, value):
        originals.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, value)

    for mixer in mixers:
        forward = _rebind_globals(
            mixer.forward.__func__,
            {"chunk_kda": delta, "fused_recurrent_kda": delta},
        )
        replace(mixer, "forward", MethodType(guard(forward), mixer))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            convolution = getattr(mixer, name)
            replace(convolution, "forward", MethodType(conv, convolution))

    def restore():
        for obj, name, existed, value in reversed(originals):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)

    return restore
