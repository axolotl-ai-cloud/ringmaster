"""Multimodal: shard the fused sequence *after* the encoder.

The vision/audio encoder runs on the short unsharded modality tokens; only the long
fused sequence entering the language backbone is CP-sharded. So a forward pre-hook on
the backbone shards ``inputs_embeds``/``position_ids`` once the encoder produced them.

    register_multimodal_sharding(model, language_model_attr="model.language_model")
"""

from __future__ import annotations

import torch

from ringmaster.runtime import get_runtime
from ringmaster.shard import shard_batch


def _resolve(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def register_multimodal_sharding(model, language_model_attr: str):
    """Shard inputs_embeds on the language backbone after the encoder runs.

    Returns the hook handle. The CP runtime must already be set up.
    """
    backbone = _resolve(model, language_model_attr)

    def pre_hook(_module, args, kwargs):
        runtime = get_runtime()
        if runtime.cp_size == 1:
            return None
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return None
        # shard_batch keys off "input_ids" shape; adapt to the embeds seq dim.
        batch = {"input_ids": embeds, **{k: v for k, v in kwargs.items() if k != "inputs_embeds"}}
        batch, _ = shard_batch(batch, cp_rank=runtime.cp_rank, cp_size=runtime.cp_size)
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = batch.pop("input_ids")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v
        return args, kwargs

    return backbone.register_forward_pre_hook(pre_hook, with_kwargs=True)
