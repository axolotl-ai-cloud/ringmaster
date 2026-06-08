"""TRL SFTTrainer + ringmaster CP e2e smoke on a REAL Qwen3.5 (gated-delta linear attn).

Unlike examples/trl_cp.py (SmolLM2, standard attention), this exercises the
Qwen3.5 gated-delta linear-attention CP path — the one that needs the sm_120
TileLang warp-spec shim. It mirrors the axolotl plugin's _wire_recurrent_layers
so the TRL and axolotl smokes run the same ringmaster wiring.

CP / TRL integration: a trainer subclass broadcasts each batch across the CP
group (all CP ranks must see the SAME sample), builds pre-shifted ``shift_labels``,
then contiguously shards input_ids/attention_mask/position_ids/shift_labels per CP
rank — the same contiguous sharding ring attention assumes. Passing ``shift_labels``
lets both the model loss and TRL's entropy metric take their CP-aware paths, so
sharded logits line up with sharded targets.

    accelerate launch --num_processes 2 qwen35_trl_cp_smoke.py
"""
import os
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import ringmaster as rm
from ringmaster.strategies.linear_attn import (
    _fla_backward_ok,
    _fla_has_cp_context,
    wrap_linear_attn_instance,
)
from ringmaster.strategies.state_passing import is_recurrent_mixer

MODEL_ID = "Qwen/Qwen3.5-4B"


class CPSFTTrainer(SFTTrainer):
    """SFTTrainer that shards each batch over a ringmaster CP group."""

    def __init__(self, *a, cp_runtime=None, **k):
        super().__init__(*a, **k)
        self._cp = cp_runtime

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rt = self._cp
        if rt is not None and rt.cp_size > 1 and inputs.get("input_ids") is not None:
            from ringmaster import broadcast_batch

            # all CP ranks must process the SAME sample (shared, shape-safe broadcast)
            broadcast_batch(inputs, rt.cp_group)

            ids = inputs["input_ids"]
            bsz, seqlen = ids.shape
            dev = ids.device
            cp = rt.cp_size
            pad = (cp - seqlen % cp) % cp

            labels = inputs.get("labels")
            if labels is None:
                labels = ids
            # pre-shift: shift_labels[t] is the target for position t
            shift = torch.full_like(labels, -100)
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
            sl = lambda t: t.chunk(cp, dim=1)[r].contiguous()
            inputs = dict(inputs)
            inputs["input_ids"] = sl(ids)
            inputs["shift_labels"] = sl(shift)
            inputs["labels"] = sl(ids)  # non-None so the model takes the loss path; shift_labels wins
            inputs["position_ids"] = sl(pos)
            if attn is not None:
                inputs["attention_mask"] = sl(attn)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


def main():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    runtime = rm.setup(
        rm.RingmasterConfig(size=world, backend=rm.Backend.RING),
        num_kv_heads=model.config.num_key_value_heads,
        device_mesh=None,
        inner_attn="flash_attention_2",
    )
    model.set_attn_implementation(runtime.attn_implementation)

    # mirror axolotl ContextParallelPlugin._wire_recurrent_layers
    text_cfg = getattr(model.config, "get_text_config", lambda: model.config)()
    conv_k = getattr(text_cfg, "linear_conv_kernel_dim", 4)
    n_linear = 0
    for module in model.modules():
        if is_recurrent_mixer(module) and hasattr(module, "chunk_gated_delta_rule"):
            wrap_linear_attn_instance(module, conv_k)
            n_linear += 1

    if rank == 0:
        native = _fla_has_cp_context() and _fla_backward_ok()
        print(
            f"[trl_cp] model={MODEL_ID} CP backend=ring size={world} "
            f"attn={runtime.attn_implementation} linear_mixers_wired={n_linear} "
            f"PATH={'native' if native else 'torch_fallback'} "
            f"(cp_context={_fla_has_cp_context()}, bwd_ok={_fla_backward_ok()}, "
            f"dispatch_disabled={os.environ.get('FLA_DISABLE_BACKEND_DISPATCH')})",
            flush=True,
        )

    texts = ["The quick brown fox jumps over the lazy dog. " * 90] * 32
    ds = Dataset.from_dict({"text": texts})

    args = SFTConfig(
        output_dir="/tmp/qwen35_trl_cp_out",
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        bf16=True,
        max_length=2048,
        report_to=[],
        save_strategy="no",
        dataset_num_proc=1,
        optim="adamw_torch",
    )
    trainer = CPSFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, cp_runtime=runtime
    )
    trainer.train()
    if rank == 0:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        print(f"[trl_cp] DONE losses={losses}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
