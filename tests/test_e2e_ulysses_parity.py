"""Gated multi-GPU parity test: Ulysses == full attention (per sequence shard).

The correctness gate for the CP redesign: a tiny causal LM run with Ulysses
sequence parallelism must produce, on each rank, logits equal to the
corresponding sequence slice of the same model run with ordinary (full) attention.

Requires >= 2 CUDA devices; skipped otherwise. Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_e2e_ulysses_parity.py
or directly (it spawns its own processes):
    python -m pytest tests/test_e2e_ulysses_parity.py
"""

import os

import pytest
import torch

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _worker(rank: int, world: int, inner_attn: str, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29588")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster.shard import shard_batch
        from transformers import LlamaConfig, LlamaForCausalLM

        flash = inner_attn.startswith("flash")
        dtype = torch.bfloat16 if flash else torch.float32
        head_dim = 32 if flash else 8  # flash needs head_dim >= 16

        torch.manual_seed(0)  # identical weights across ranks (deterministic init)
        cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=8 * head_dim,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
        )
        model = LlamaForCausalLM(cfg).to(rank).to(dtype).eval()

        seq = 32
        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (1, seq), device=rank)
        position_ids = torch.arange(seq, device=rank).unsqueeze(0)

        # Reference: ordinary full attention with the same kernel, whole sequence.
        model.set_attn_implementation(inner_attn)
        with torch.no_grad():
            ref = model(input_ids=input_ids, position_ids=position_ids).logits

        # Ulysses: wrap the kernel, shard the sequence, forward this rank's shard.
        runtime = rm.setup(
            rm.RingmasterConfig(size=world, backend=rm.Backend.ULYSSES),
            num_kv_heads=8,
            device_mesh=None,
            inner_attn=inner_attn,
        )
        model.set_attn_implementation(runtime.attn_implementation)

        batch = {"input_ids": input_ids.clone(), "position_ids": position_ids.clone()}
        batch, _ = shard_batch(batch, cp_rank=runtime.cp_rank, cp_size=runtime.cp_size)
        with torch.no_grad(), rm.profile_comms() as stats:
            out = model(**batch).logits

        local = seq // world
        ref_slice = ref[:, rank * local : (rank + 1) * local]
        max_err = (out.float() - ref_slice.float()).abs().max().item()
        # MHA fuses q/k/v: 2 layers x 2 all-to-all (qkv, out) = 4 collectives
        n_collectives = sum(stats.count.values())
        out_q.put((rank, max_err, n_collectives))
        if rank == 0:
            print("\n" + stats.report())
        rm.teardown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not CUDA_OK, reason="needs >= 2 CUDA devices")
@pytest.mark.parametrize(
    "inner_attn,tol", [("sdpa", 2e-3), ("flash_attention_2", 3e-2)]
)
def test_ulysses_matches_full_attention(inner_attn, tol):
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, inner_attn, out_q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=120)

    for rank, max_err, n_collectives in results:
        assert max_err < tol, f"rank {rank} Ulysses({inner_attn}) err {max_err}"
        assert n_collectives == 4, f"rank {rank} expected 4 collectives, got {n_collectives}"
