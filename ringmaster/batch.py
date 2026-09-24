"""Shape- and dtype-safe batch replication within context-parallel groups."""

import torch
import torch.distributed as dist


def broadcast_batch(kwargs: dict, group, src: int | None = None) -> None:
    """Broadcast every sequence tensor in ``kwargs`` from ``src`` across ``group``.

    Shape-safe: agree on the source shape first, else an in-place broadcast writes
    the payload into a smaller local buffer (variable-length samples) → OOB.
    """
    if group is None or dist.get_world_size(group) == 1:
        return
    if src is None:
        src = dist.get_process_group_ranks(group)[0]
    is_src = dist.get_rank() == src
    metadata = [
        [(key, tuple(val.shape), val.dtype, val.device.type)
         for key, val in kwargs.items()
         if isinstance(val, torch.Tensor) and val.dim() > 0]
        if is_src else None
    ]
    dist.broadcast_object_list(metadata, src=src, group=group)
    keys = {item[0] for item in metadata[0]}
    for key, val in list(kwargs.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 0 and key not in keys:
            del kwargs[key]
    for key, shape, dtype, device_type in metadata[0]:
        val = kwargs.get(key)
        if is_src:
            tensor = val.contiguous()
        else:
            device = val.device if isinstance(val, torch.Tensor) and val.device.type == device_type else torch.device(device_type)
            tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(tensor, src=src, group=group)
        kwargs[key] = tensor
