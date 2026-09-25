"""Framework-agnostic CP data orchestration shared by the axolotl plugin, trl
trainer and raw scripts: broadcast the batch so all ranks shard the same sample,
shard the sequence dim, normalize num_items / correct eval loss, optionally gather
outputs. Attention is the backend's job (ring/ulysses + linear-attn/mamba wraps).
"""

from __future__ import annotations

import inspect

import torch
import torch.distributed as dist

from ringmaster.cp_collectives import seq_gather_cp
from ringmaster.loss import correct_eval_loss, global_num_items_in_batch
from ringmaster.runtime import maybe_runtime
from ringmaster.shard import shard_batch, varlen_meta


from ringmaster.batch import broadcast_batch


class ContextParallelContextManager:
    """Install CP data hooks on ``models``. Use as a context manager or via
    :meth:`install` / :meth:`remove` for a persistent (plugin-style) lifecycle."""

    def __init__(
        self,
        models,
        cp_group,
        *,
        gradient_accumulation_steps: int = 1,
        gather_outputs: bool = False,
        load_balance: str = "contiguous",
    ):
        self.models = list(models)
        self.cp_group = cp_group
        self.grad_accum = int(gradient_accumulation_steps or 1)
        self.gather_outputs = gather_outputs
        self.load_balance = load_balance
        self.cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
        self.cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0
        self._handles: list = []
        self._local_valid = None
        self._pad_len = 0
        self._orig_seq_len = 0

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def install(self):
        for model in self.models:
            forward_params = list(inspect.signature(model.forward).parameters.keys())
            self._handles.append(
                model.register_forward_pre_hook(
                    self._make_pre_hook(forward_params), with_kwargs=True
                )
            )
            self._handles.append(model.register_forward_hook(self._post_hook))
        return self._handles

    def _make_pre_hook(self, forward_params):
        def pre_hook(module, args, kwargs):
            kwargs = dict(kwargs)
            for i, arg in enumerate(args):
                if i < len(forward_params):
                    kwargs[forward_params[i]] = arg
            remaining = args[len(forward_params) :]

            ids = kwargs.get("input_ids")
            if ids is None or self.cp_size == 1:
                return remaining, kwargs

            broadcast_batch(kwargs, self.cp_group)
            ids = kwargs["input_ids"]
            mask = kwargs.get("attention_mask")
            if mask is not None:
                if (
                    mask.ndim != 2
                    or torch.any(mask < 0)
                    or (
                        mask.is_floating_point()
                        and torch.any(~torch.isfinite(mask) | (mask != mask.round()))
                    )
                    or torch.any((mask[:, 1:] > 0) & (mask[:, :-1] == 0))
                ):
                    raise ValueError(
                        "Context parallelism requires a right-padded causal 2D mask"
                    )
                if torch.any(mask > 1) and kwargs.get("position_ids") is None:
                    positions = torch.arange(
                        mask.shape[1], device=mask.device
                    ).expand_as(mask)
                    starts = torch.ones_like(mask, dtype=torch.bool)
                    starts[:, 1:] = mask[:, 1:] != mask[:, :-1]
                    offsets = torch.where(starts, positions, 0).cummax(dim=1).values
                    kwargs["position_ids"] = positions - offsets
                kwargs.pop("attention_mask")
            if kwargs.get("position_ids") is None:
                cu = kwargs.get("cu_seqlens")
                if cu is None:
                    cu = kwargs.get("cu_seq_lens_q")
                if cu is not None:
                    cu = cu.to(device=ids.device, dtype=torch.long)
                    if (
                        cu.ndim != 1
                        or cu.numel() < 2
                        or int(cu[0]) != 0
                        or int(cu[-1]) != ids.numel()
                        or torch.any(cu[1:] <= cu[:-1])
                    ):
                        raise ValueError("Invalid global packed document boundaries")
                    positions = torch.arange(ids.numel(), device=ids.device)
                    documents = torch.bucketize(positions, cu[1:], right=True)
                    kwargs["position_ids"] = (positions - cu[documents]).reshape_as(ids)
            # Global collator metadata cannot describe a local shard.
            for key in (
                "cu_seq_lens_q",
                "cu_seq_lens_k",
                "max_length_q",
                "max_length_k",
                "cu_seqlens",
                "max_seqlen",
            ):
                kwargs.pop(key, None)
            global_pos = kwargs.get("position_ids")  # full sequence, before sharding
            kwargs, info = shard_batch(
                kwargs,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
                load_balance=self.load_balance,
            )
            self._orig_seq_len, self._pad_len = info.original_seq_len, info.pad_len
            # stash packed-sequence cu_seqlens for the Ulysses varlen path (None clears
            # any stale value from the previous step).
            rt = maybe_runtime()
            if rt is not None:
                rt.varlen = varlen_meta(
                    global_pos, info.original_seq_len + info.pad_len
                )

            # count valid tokens from shift_labels (what the model's loss uses)
            count_labels = kwargs.get("shift_labels")
            if count_labels is None:
                count_labels = kwargs.get("labels")
            if count_labels is not None:
                if module.training and kwargs.get("num_items_in_batch") is not None:
                    kwargs["num_items_in_batch"] = global_num_items_in_batch(
                        count_labels, self.cp_group, self.grad_accum
                    )
                if not module.training:
                    self._local_valid = (count_labels != -100).sum().float()
                    kwargs.pop("num_items_in_batch", None)
            return remaining, kwargs

        return pre_hook

    def _post_hook(self, module, inputs, output):
        if self.cp_size > 1 and self.gather_outputs:
            # Reassemble the sequence-sharded outputs (assumes contiguous layout —
            # GRPO/EBFT use it; zigzag would need the inverse permutation).
            local_len = (self._orig_seq_len + self._pad_len) // self.cp_size
            for key, val in list(output.items()):
                if (
                    isinstance(val, torch.Tensor)
                    and val.dim() > 1
                    and val.size(1) == local_len
                ):
                    gathered = seq_gather_cp(val, self.cp_group)
                    if self._pad_len:
                        gathered = gathered[:, : self._orig_seq_len].contiguous()
                    output[key] = gathered
        if self._local_valid is not None and getattr(output, "loss", None) is not None:
            output["loss"] = correct_eval_loss(
                output.loss, self._local_valid, self.cp_group
            )
        self._local_valid = None
        return output
