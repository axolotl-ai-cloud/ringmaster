"""TRL SFTTrainer + ringmaster context parallelism + FSDP2 — one script, any model.

Context parallelism (CP) splits each sample's *sequence* across the CP group and ring
attention computes full-sequence attention across it; FSDP2 shards the *parameters*
and reduces gradients over the same (dp_shard x cp) mesh — which SUMS the CP-partial
gradients back, exactly what ring attention needs.

How CP + FSDP2 coexist in bare TRL (the axolotl trick, no transformers/accelerate
fork required):

  1. Pass ``parallelism_config=ParallelismConfig(cp_size=C, dp_shard_size=D)`` so
     accelerate builds a ``dp_shard x cp`` device mesh. ``cp`` is a non-DP dim, and
     FSDP2 shards/reduces over the flattened ``dp_shard_cp`` mesh (grads sum over cp).
  2. Override ``_prepare_context_parallel_inputs`` to a no-op so transformers' built-in
     (torch, SDPA-only) context parallel is bypassed — ringmaster's ring attention
     (set via ``set_attn_implementation``) owns CP instead.
  3. Shard the batch ourselves in ``compute_loss`` (TRL computes its loss/metrics
     OUTSIDE the model forward, so a forward pre-hook would desync them): pre-shift
     ``shift_labels`` and rescale ``num_items_in_batch`` to the global token count so
     the per-rank loss matches FSDP2's AVG gradient reduction over the CP dim.

    # Llama-3.1-8B, 8-way CP (dp_shard auto = world / cp)
    accelerate launch --num_processes 8 --mixed_precision bf16 trl_cp_fsdp2.py \
        --model meta-llama/Llama-3.1-8B --seq-len 32768 --cp-size 8

    # 16 GPUs as 2-way data-parallel x 8-way CP
    accelerate launch --num_processes 16 --mixed_precision bf16 trl_cp_fsdp2.py \
        --model meta-llama/Llama-3.1-8B --seq-len 32768 --cp-size 8

    # Qwen3.5 (gated-delta linear attention) — needs the fla sm_120 shim on Blackwell
    accelerate launch --num_processes 8 --mixed_precision bf16 trl_cp_fsdp2.py \
        --model Qwen/Qwen3.5-4B --seq-len 32768 --cp-size 8

    # Nemotron-H (hybrid Mamba2 + attention) — pip install mamba-ssm causal-conv1d
    accelerate launch --num_processes 8 --mixed_precision bf16 trl_cp_fsdp2.py \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --seq-len 16384 --cp-size 8
"""

import argparse
import contextlib

import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import ringmaster as rm
from ringmaster import broadcast_batch, wire_recurrent_layers
from ringmaster.adapters.accelerate import setup_from_accelerate
from ringmaster.loss import global_num_items_in_batch


class CPSFTTrainer(SFTTrainer):
    """SFTTrainer that lets ringmaster own context parallelism.

    ``_prepare_context_parallel_inputs`` is overridden to a no-op so transformers'
    built-in torch/SDPA context parallel is bypassed; ``compute_loss`` shards each
    sample's sequence over the ringmaster CP group (with pre-shifted ``shift_labels``)
    and rescales ``num_items_in_batch`` to match FSDP2's AVG gradient reduction.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cp = None  # ringmaster runtime; set after the accelerator mesh exists

    def _prepare_context_parallel_inputs(self, model, inputs):
        if self._cp is not None and self._cp.cp_size > 1:
            return contextlib.nullcontext, inputs  # ringmaster ring attention owns CP
        return super()._prepare_context_parallel_inputs(model, inputs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        rt = self._cp
        if rt is not None and rt.cp_size > 1 and inputs.get("input_ids") is not None:
            broadcast_batch(inputs, rt.cp_group)  # every CP rank: the SAME sample
            ids = inputs["input_ids"]
            bsz, seqlen = ids.shape
            dev = ids.device
            cp = rt.cp_size
            pad = (cp - seqlen % cp) % cp

            labels = inputs.get("labels")
            labels = ids if labels is None else labels
            shift = torch.full_like(
                labels, -100
            )  # shift[t] is the target for position t
            shift[:, :-1] = labels[:, 1:]
            attn = inputs.get("attention_mask")
            pos = torch.arange(seqlen, device=dev).unsqueeze(0).expand(bsz, -1)
            if pad:
                ids = torch.cat([ids, ids.new_zeros(bsz, pad)], 1)
                shift = torch.cat([shift, shift.new_full((bsz, pad), -100)], 1)
                pos = torch.cat([pos, pos.new_zeros(bsz, pad)], 1)
                if attn is not None:
                    attn = torch.cat([attn, attn.new_zeros(bsz, pad)], 1)

            r = rt.cp_rank

            def sl(t):
                return t.chunk(cp, dim=1)[r].contiguous()

            inputs = dict(inputs)
            inputs["input_ids"] = sl(ids)
            inputs["shift_labels"] = sl(shift)
            inputs["labels"] = sl(
                ids
            )  # non-None so the model takes its loss path; shift_labels wins
            inputs["position_ids"] = sl(pos)
            if attn is not None:
                inputs["attention_mask"] = sl(attn)
            if num_items_in_batch is not None:
                num_items_in_batch = global_num_items_in_batch(
                    inputs["shift_labels"],
                    rt.cp_group,
                    self.args.gradient_accumulation_steps,
                )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def maybe_apply_sm120_shims():
    """Best-effort: enable fla's sm_120 (Blackwell) gated-delta shim and prefer the
    locally-installed mamba-ssm/causal-conv1d kernels (Qwen3.5/Nemotron on cu13+sm_120).
    No-op elsewhere. Must run before the model loads."""
    try:
        from ringmaster.strategies.linear_attn import _apply_fla_sm120_shim

        _apply_fla_sm120_shim()
    except Exception:
        pass
    try:
        from ringmaster.strategies.mamba import (
            ensure_causal_conv1d_cuda_export,
            prefer_local_mamba_kernels,
        )

        prefer_local_mamba_kernels()
        ensure_causal_conv1d_cuda_export()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument(
        "--cp-size", type=int, default=0, help="CP degree (default: whole world)"
    )
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--no-fsdp", action="store_true", help="disable FSDP2 (pure CP)")
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    cp = args.cp_size or world
    assert world % cp == 0, f"world ({world}) must be divisible by cp_size ({cp})"
    dp_shard = world // cp

    maybe_apply_sm120_shims()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    text = "The quick brown fox jumps over the lazy dog. " * (args.seq_len // 10)
    ds = Dataset.from_dict({"text": [text] * 32})

    # Non-retaining long-context loss. Liger fused-linear-CE (custom Triton fwd+bwd)
    # materializes no logits and frees each chunk's grad inline; trl's chunked_nll
    # checkpoints+sums per-chunk losses, so backward holds ALL chunks' log_p + the
    # entropy metric at once (~2x full fp32 logits — the long-context killer, ~16 GB
    # more than Liger at 512k). Liger honors our pre-shifted shift_labels (it skips its
    # internal shift when shift_labels is given), so the CP chunk boundary stays exact.
    # Liger only patches some arches; fall back to chunked_nll where it doesn't (e.g.
    # Nemotron-H), since a silent no-op would drop to full-logits CE and OOM.
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    use_liger = model.config.model_type in MODEL_TYPE_TO_APPLY_LIGER_FN
    loss_kwargs = (
        {"use_liger_kernel": True} if use_liger else {"loss_type": "chunked_nll"}
    )

    # accelerate builds the dp_shard x cp mesh; FSDP2 shards/reduces over dp_shard_cp.
    pc = ParallelismConfig(cp_size=cp, dp_shard_size=dp_shard)
    fsdp_kwargs = {}
    if not args.no_fsdp:
        present = {type(m).__name__ for m in model.modules()}
        layers = [
            c for c in (getattr(model, "_no_split_modules", None) or []) if c in present
        ]
        fsdp_kwargs = dict(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "fsdp_version": 2,
                "transformer_layer_cls_to_wrap": layers,
                "reshard_after_forward": True,
                "state_dict_type": "SHARDED_STATE_DICT",
            },
        )
    sft = SFTConfig(
        output_dir="/tmp/trl_cp_fsdp2_out",
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        logging_steps=1,
        bf16=True,
        max_length=args.seq_len,
        report_to=[],
        save_strategy="no",
        dataset_num_proc=1,
        optim="adamw_torch",
        parallelism_config=pc,
        **loss_kwargs,
        **fsdp_kwargs,
    )
    trainer = CPSFTTrainer(
        model=model, args=sft, train_dataset=ds, processing_class=tok
    )

    # Wire ringmaster from the accelerator's device mesh (reads the "cp" dim), swap in
    # ring attention, and wire any recurrent (Mamba/linear-attention) layers for CP.
    runtime = setup_from_accelerate(
        trainer.accelerator,
        rm.RingmasterConfig(size=cp, backend=rm.Backend.RING),
        num_kv_heads=model.config.num_key_value_heads,
        cp_dim="cp",
    )
    model.set_attn_implementation(runtime.attn_implementation)
    text_cfg = getattr(model.config, "get_text_config", lambda: model.config)()
    text_cfg.use_cache = False
    wiring = wire_recurrent_layers(model)
    trainer._cp = runtime

    if rank == 0:
        print(
            f"[trl_cp_fsdp2] model={args.model} cp={cp} dp_shard={dp_shard} "
            f"loss={'liger_flce' if use_liger else 'chunked_nll'} "
            f"attn={runtime.attn_implementation} fsdp2={not args.no_fsdp} "
            f"recurrent_wired={bool(wiring)} (linear_attn={wiring.linear_attn_mixers}, "
            f"mamba={list(wiring.mamba_modules)})",
            flush=True,
        )
    trainer.train()
    if rank == 0:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        print(f"[trl_cp_fsdp2] DONE losses={losses}", flush=True)
    wiring.restore()
    rm.teardown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
