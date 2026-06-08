"""Gated multi-GPU parity: ring attention == full attention (per shard).

Shards both inputs and reference with the runtime's load_balance (zigzag reorders
tokens) and compares this rank's slice — exactly how the axolotl plugin wires it,
across providers and layouts. Requires >= 2 CUDA devices."""

import os

import pytest
import torch

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2

# (provider, load_balance): provider drives the ring loop's block kernel and only
# matters for the contiguous ring (none); head_tail/distflash route to their own
# flash kernels and ignore the provider.
CASES = [
    ("hf_kernels", "none"),
    ("torch_native", "none"),
    ("hf_kernels", "head_tail"),
    ("hf_kernels", "distflash"),
]


def _worker(rank: int, world: int, provider: str, load_balance: str, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29590")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster.shard import shard_batch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=256,
        )
        model = LlamaForCausalLM(cfg).to(rank).to(torch.bfloat16).eval()

        seq = 64
        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (1, seq), device=rank)
        position_ids = torch.arange(seq, device=rank).unsqueeze(0)

        # Reference: full flash attention over the whole sequence.
        model.set_attn_implementation("flash_attention_2")
        with torch.no_grad():
            ref = model(input_ids=input_ids, position_ids=position_ids).logits.float()

        ring_impl = (
            rm.RingImpl.HF_KERNELS if provider == "hf_kernels" else rm.RingImpl.TORCH_NATIVE
        )
        runtime = rm.setup(
            rm.RingmasterConfig(
                size=world, backend=rm.Backend.RING, ring_impl=ring_impl,
                load_balance=rm.LoadBalance(load_balance),
            ),
            num_kv_heads=8,
            device_mesh=None,
            inner_attn="flash_attention_2",
        )
        model.set_attn_implementation(runtime.attn_implementation)

        # Shard inputs via the DEFAULT (resolved from the runtime) — the production
        # path; guards the config/shard single-source-of-truth (a contiguous shard
        # under a head_tail config would silently corrupt). Reference uses the same
        # explicit layout so the per-rank compare holds whatever the layout.
        batch = {"input_ids": input_ids.clone(), "position_ids": position_ids.clone()}
        batch, _ = shard_batch(batch, cp_rank=runtime.cp_rank, cp_size=runtime.cp_size)
        with torch.no_grad():
            out = model(**batch).logits.float()

        ref_batch = {"input_ids": input_ids.clone(), "logits": ref}
        ref_batch, _ = shard_batch(ref_batch, cp_rank=runtime.cp_rank,
                                   cp_size=runtime.cp_size, load_balance=load_balance)
        err = (out - ref_batch["logits"]).abs().max().item()
        out_q.put((rank, provider, load_balance, err))
        rm.teardown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not CUDA_OK, reason="needs >= 2 CUDA devices")
@pytest.mark.parametrize("provider,load_balance", CASES)
def test_ring_matches_full_attention(provider, load_balance):
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, provider, load_balance, out_q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=180) for _ in range(world)]
    for p in procs:
        p.join(timeout=180)

    for rank, prov, lb, err in results:
        assert err < 3e-2, f"rank {rank} ring({prov},{lb}) vs full attention max err {err}"
