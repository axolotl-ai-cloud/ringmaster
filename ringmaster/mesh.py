"""Device-mesh wiring: resolve the CP process group(s) and build a CPRuntime.

A host mesh's ``cp`` dim is interpreted as ``ulysses_size x ring_size``. Ulysses-
or Ring-only just use the cp group; the 2D USP split must be built at mesh
construction (``cp_ulysses``/``cp_ring`` dims), so it can't be done post-hoc here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from ringmaster.compat import require_torch
from ringmaster.config import RingmasterConfig
from ringmaster.runtime import CPRuntime

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh


def intra_node_size() -> int:
    """Best-effort GPUs-per-node, used by the auto-selector to bound Ulysses."""
    for var in ("LOCAL_WORLD_SIZE", "NPROC_PER_NODE", "SLURM_NTASKS_PER_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:  # pragma: no cover - defensive
        pass
    return 1


def _cp_group_from_mesh(device_mesh: "DeviceMesh", cp_dim: str):
    try:
        submesh = device_mesh[cp_dim]
    except (KeyError, IndexError) as e:
        names = getattr(device_mesh, "mesh_dim_names", None)
        raise ValueError(
            f"cp dim '{cp_dim}' not in device_mesh (dims: {names})"
        ) from e
    return submesh.get_group()


def build_runtime(
    config: RingmasterConfig,
    *,
    device_mesh: Optional["DeviceMesh"] = None,
    cp_dim: str = "cp",
    device_type: str = "cuda",
) -> CPRuntime:
    """Resolve process groups and return a :class:`CPRuntime`. ``config`` must be
    normalized first (``RingmasterConfig.normalize`` — head-count is a model property)."""
    if not config.enabled:
        return CPRuntime(config=config)

    require_torch()
    import torch.distributed as dist

    dims = set(device_mesh.mesh_dim_names or ()) if device_mesh is not None else set()
    has_cp = device_mesh is not None and (
        cp_dim in dims or {"cp_ulysses", "cp_ring"} <= dims
    )
    if has_cp and cp_dim in dims:
        cp_group = _cp_group_from_mesh(device_mesh, cp_dim)
    elif has_cp:  # composed USP: cp dim already split into cp_ulysses x cp_ring
        cp_group = dist.group.WORLD
    else:
        # No cp dim in the host mesh (e.g. plain DDP) — build a standalone CP group.
        if device_mesh is not None:
            LOG.info(
                "device mesh has no '%s' dim (dims=%s); building a standalone CP group "
                "over the world.",
                cp_dim,
                device_mesh.mesh_dim_names,
            )
        device_mesh = None
        from torch.distributed.device_mesh import init_device_mesh

        if config.ulysses_size > 1 and config.ring_size > 1:
            cp_group = dist.group.WORLD  # _split_usp_groups builds the 2D mesh
        else:
            mesh = init_device_mesh(device_type, (config.size,), mesh_dim_names=(cp_dim,))
            cp_group = mesh[cp_dim].get_group()

    if dist.get_world_size(cp_group) != config.size:
        raise ValueError(
            f"cp group size {dist.get_world_size(cp_group)} != configured size {config.size}"
        )

    runtime = CPRuntime(config=config, cp_group=cp_group)

    if config.ring_size == 1:
        runtime.ulysses_group = cp_group
        runtime.ring_group = None
    elif config.ulysses_size == 1:
        runtime.ring_group = cp_group
        runtime.ulysses_group = None
    else:
        _split_usp_groups(runtime, config, device_mesh, cp_dim, device_type)

    return runtime


def _split_usp_groups(runtime, config, device_mesh, cp_dim, device_type):
    """Populate ulysses_group + ring_group for hybrid USP — standalone builds a 2D
    ``(ring, ulysses)`` mesh; composed reads ``cp_ulysses``/``cp_ring`` from the host."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if device_mesh is not None and {"cp_ulysses", "cp_ring"} <= set(
        device_mesh.mesh_dim_names or ()
    ):
        runtime.ulysses_group = device_mesh["cp_ulysses"].get_group()
        runtime.ring_group = device_mesh["cp_ring"].get_group()
        return

    if device_mesh is not None:
        raise NotImplementedError(
            "Composed USP needs the host device mesh built with separate 'cp_ulysses' "
            "and 'cp_ring' dims (split the cp dim at mesh construction). Found dims: "
            f"{device_mesh.mesh_dim_names}."
        )

    # Standalone: ulysses is the inner (contiguous) dim, ring the outer.
    mesh = init_device_mesh(
        device_type,
        (config.ring_size, config.ulysses_size),
        mesh_dim_names=("cp_ring", "cp_ulysses"),
    )
    runtime.ulysses_group = mesh["cp_ulysses"].get_group()
    runtime.ring_group = mesh["cp_ring"].get_group()
    runtime.cp_group = dist.group.WORLD
