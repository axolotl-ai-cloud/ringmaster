"""TRL SFTTrainer + ringmaster context parallelism (ring attention).

Each CP rank processes the same batch's sequence shard; ring attention computes
full-sequence attention across the CP group. Standard-attention model (no fla
needed). Launch on 2 GPUs:
    accelerate launch --num_processes 2 examples/trl_cp.py
"""
import os
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import ringmaster as rm
from ringmaster.shard import shard_batch


def main():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    model_id = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    # --- ringmaster CP: ring attention over the world as the CP group ---
    runtime = rm.setup(
        rm.RingmasterConfig(size=world, backend=rm.Backend.RING),
        num_kv_heads=model.config.num_key_value_heads,
        device_mesh=None,  # standalone -> CP group == world
        inner_attn="flash_attention_2",
    )
    model.set_attn_implementation(runtime.attn_implementation)

    # All CP ranks must see the SAME batch, then shard its sequence.
    def cp_pre_hook(_m, args, kwargs):
        if runtime.cp_size == 1 or "input_ids" not in kwargs:
            return None
        src = dist.get_process_group_ranks(runtime.cp_group)[0]
        for k, v in list(kwargs.items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                t = v.contiguous()
                dist.broadcast(t, src=src, group=runtime.cp_group)
                kwargs[k] = t
        kwargs, _ = shard_batch(kwargs, cp_rank=runtime.cp_rank, cp_size=runtime.cp_size)
        return args, kwargs

    model.register_forward_pre_hook(cp_pre_hook, with_kwargs=True)

    # --- tiny dataset ---
    texts = ["The quick brown fox jumps over the lazy dog. " * 40] * 64
    ds = Dataset.from_dict({"text": texts})

    args = SFTConfig(
        output_dir="/tmp/trl_cp_out",
        max_steps=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        bf16=True,
        max_length=2048,
        report_to=[],
        save_strategy="no",
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    if rank == 0:
        print(f"[trl_cp] CP enabled: backend=ring size={world} attn={runtime.attn_implementation}", flush=True)
    trainer.train()
    if rank == 0:
        print("[trl_cp] DONE", flush=True)


if __name__ == "__main__":
    main()
