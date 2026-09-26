"""Instance-local state passing for FLA Mamba mixers."""

from types import MethodType, SimpleNamespace


def fla_mamba_mixers(model):
    return [
        module
        for module in model.modules()
        if any(
            cls.__module__ in ("fla.layers.mamba", "fla.layers.mamba2")
            and cls.__name__ in ("Mamba", "Mamba2")
            for cls in type(module).__mro__
        )
    ]


def _scan1(original, group):
    import torch
    import torch.distributed as dist
    from ringmaster.cp_collectives import all_gather_cp
    from ringmaster.runtime import maybe_runtime
    from ringmaster.ring.varlen_blocks import doc_ids_from_cu

    def scan(u, dt, A, B, C, D, z, bias):
        batch, channels, length = u.shape
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        runtime = maybe_runtime()
        documents = None
        if runtime is not None and runtime.varlen is not None:
            documents = doc_ids_from_cu(runtime.varlen[0], u.device).reshape(
                batch, length * world
            )
        effective_dt = torch.nn.functional.softplus(
            dt.float() + bias.float()[None, :, None]
        )
        cumulative = effective_dt.cumsum(-1)
        local_docs = (
            documents[:, rank * length : (rank + 1) * length]
            if documents is not None
            else None
        )
        if local_docs is None:
            output = original(
                u.contiguous(),
                dt.contiguous(),
                A,
                B.contiguous(),
                C.contiguous(),
                D,
                z.contiguous(),
                bias,
                delta_softplus=True,
            )
        else:
            rows = []
            for row in range(batch):
                starts = (
                    [0]
                    + (
                        (local_docs[row, 1:] != local_docs[row, :-1])
                        .nonzero()
                        .flatten()
                        + 1
                    ).tolist()
                    + [length]
                )
                rows.append(
                    torch.cat(
                        [
                            original(
                                u[row : row + 1, :, start:end].contiguous(),
                                dt[row : row + 1, :, start:end].contiguous(),
                                A,
                                B[row : row + 1, :, start:end].contiguous(),
                                C[row : row + 1, :, start:end].contiguous(),
                                D,
                                z[row : row + 1, :, start:end].contiguous(),
                                bias,
                                delta_softplus=True,
                            )
                            for start, end in zip(starts[:-1], starts[1:])
                        ],
                        dim=-1,
                    )
                )
            output = torch.cat(rows)
        final = u.new_zeros(batch, channels, A.shape[-1], dtype=torch.float32)
        for start in range(0, length, 128):
            end = min(start + 128, length)
            decay = (
                (cumulative[:, :, -1:] - cumulative[:, :, start:end])[:, :, :, None]
                * A.float()[None, :, None, :]
            ).exp()
            contributions = effective_dt[:, :, start:end] * u[:, :, start:end].float()
            if local_docs is not None:
                contributions = (
                    contributions
                    * (local_docs[:, start:end] == local_docs[:, -1:])[:, None, :]
                )
            final = final + torch.einsum(
                "bdtn,bdt,bnt->bdn", decay, contributions, B[:, :, start:end].float()
            )
        decay = (cumulative[:, :, -1, None] * A.float()).exp()
        continuation = None
        if local_docs is not None:
            previous = documents[:, rank * length - 1] if rank else local_docs[:, 0] - 1
            continuation = local_docs == previous[:, None]
            decay = decay * continuation.all(-1)[:, None, None]
        packed = torch.stack((decay, final), dim=0)
        states = all_gather_cp(packed, group)
        entering = torch.zeros_like(final)
        for predecessor in range(rank):
            entering = states[predecessor, 0] * entering + states[predecessor, 1]
        parts = []
        for start in range(0, length, 128):
            end = min(start + 128, length)
            transitions = (
                cumulative[:, :, start:end, None] * A.float()[None, :, None, :]
            ).exp()
            correction = torch.einsum(
                "bdtn,bdn,bnt->bdt", transitions, entering, C[:, :, start:end].float()
            )
            if continuation is not None:
                correction = correction * continuation[:, None, start:end]
            correction = correction * torch.nn.functional.silu(
                z[:, :, start:end].float()
            )
            parts.append(output[:, :, start:end] + correction.to(output.dtype))
        return torch.cat(parts, dim=-1) + (states.sum() * 0).to(output.dtype)

    return scan


def wire_fla_mamba(mixers, group):
    import torch
    from ringmaster.mamba import _bindings

    originals = []

    def restore():
        while originals:
            mixer, existed, forward = originals.pop()
            if existed:
                mixer.forward = forward
            else:
                del mixer.forward

    def adapted(mixer):
        raw = mixer.cuda_kernels_forward.__func__
        namespace = raw.__globals__
        scan2 = namespace.get("mamba_chunk_scan_combined")
        conv = mixer.causal_conv1d_fn
        if mixer.backend != "cuda" or conv is None:
            raise ValueError("FLA Mamba CP requires loaded CUDA convolution kernels")
        bindings = _bindings(
            SimpleNamespace(
                __globals__={
                    "mamba2_chunk_scan": scan2,
                    "causal_conv1d_fn": conv,
                }
            ),
            group,
        )
        first_generation = "selective_scan_fn" in namespace
        selected_scan = (
            namespace.get("selective_scan_fn") if first_generation else scan2
        )
        if not callable(selected_scan):
            raise ValueError("FLA Mamba CP requires loaded Mamba scan kernels")
        scan1 = (
            _scan1(namespace["selective_scan_fn"], group) if first_generation else None
        )

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        ):
            if past_key_values is not None or use_cache:
                raise ValueError("FLA Mamba CP requires use_cache=False")
            projected = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())
            if first_generation:
                x, gate = projected.transpose(1, 2).chunk(2, dim=1)
                x = bindings["causal_conv1d_fn"](
                    x,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                )
                params = self.x_proj(x.transpose(1, 2))
                dt, B, C = params.split(
                    [self.dt_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
                )
                dt = self.dt_proj.weight @ dt.transpose(1, 2)
                output = scan1(
                    x,
                    dt,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    self.dt_proj.bias.float(),
                ).transpose(1, 2)
            else:
                batch, length, _ = projected.shape
                group_state = self.n_groups * self.ssm_state_size
                gate, xbc, dt = projected.split(
                    [
                        self.intermediate_size,
                        self.intermediate_size + 2 * group_state,
                        self.num_heads,
                    ],
                    dim=-1,
                )
                xbc = bindings["causal_conv1d_fn"](
                    xbc.transpose(1, 2).contiguous(),
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
                x, B, C = xbc.split(
                    [self.intermediate_size, group_state, group_state], dim=-1
                )
                output = bindings["mamba2_chunk_scan"](
                    x.reshape(batch, length, self.num_heads, self.head_dim),
                    dt,
                    A,
                    B.reshape(batch, length, self.n_groups, self.ssm_state_size),
                    C.reshape(batch, length, self.n_groups, self.ssm_state_size),
                    self.chunk_size,
                    D=self._get_D(),
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    dt_limit=self.dt_limit,
                ).reshape(batch, length, self.intermediate_size)
                output = (
                    self.norm(output, gate)
                    if self.rmsnorm
                    else output * torch.nn.functional.silu(gate)
                )
            return self.out_proj(output), None, None

        return MethodType(forward, mixer)

    try:
        for mixer in mixers:
            forward = adapted(mixer)
            originals.append(
                (mixer, "forward" in vars(mixer), vars(mixer).get("forward"))
            )
            mixer.forward = forward
    except Exception:
        restore()
        raise
    return restore
