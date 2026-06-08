"""Gated multi-GPU parity: packed-sequence (varlen) CP == single-GPU flash varlen.

A tiny causal LM on a PACKED batch (b=1, position_ids resetting per document) under
CP must produce, per shard, logits equal to the same model run on one GPU with flash
varlen attention. Covers the Ulysses flash-varlen path and the Ring document-masked
path (USP/zigzag/distflash route through the same masked path). GQA. Requires >= 2
CUDA devices; run directly (spawns its own processes):
    python -m pytest tests/test_varlen_gpu.py
"""

import os

import pytest
import torch

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2

# (doc_lengths) — each pack sums to seq_len, seq_len divisible by world (2).
PACKS = [[24, 40], [16, 16, 32], [10, 30, 24], [64]]


def _worker(rank, world, backend_name, doc_lengths, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29612")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster.shard import shard_batch, varlen_meta
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=256, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
            max_position_embeddings=512,
        )
        model = LlamaForCausalLM(cfg).to(rank).to(torch.bfloat16).eval()
        model.set_attn_implementation("flash_attention_2")

        seq = sum(doc_lengths)
        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (1, seq), device=rank)
        # packed position_ids: reset to 0 at each document boundary
        pos = torch.cat([torch.arange(n, device=rank) for n in doc_lengths]).unsqueeze(0)

        # Reference: single GPU, flash varlen from packed position_ids.
        with torch.no_grad():
            ref = model(input_ids=input_ids, position_ids=pos).logits.float()

        backend = rm.Backend.ULYSSES if backend_name == "ulysses" else rm.Backend.RING
        lb = rm.LoadBalance.DISTFLASH if backend_name == "distflash" else rm.LoadBalance.NONE
        runtime = rm.setup(
            rm.RingmasterConfig(size=world, backend=backend, load_balance=lb),
            num_kv_heads=cfg.num_key_value_heads, device_mesh=None,
            inner_attn="flash_attention_2",
        )
        model.set_attn_implementation(runtime.attn_implementation)

        batch = {"input_ids": input_ids.clone(), "position_ids": pos.clone()}
        batch, info = shard_batch(batch, cp_rank=runtime.cp_rank, cp_size=runtime.cp_size,
                                  load_balance="contiguous")
        runtime.varlen = varlen_meta(pos, info.original_seq_len + info.pad_len)
        with torch.no_grad():
            out = model(**batch).logits.float()

        local = seq // world
        err = (out - ref[:, rank * local:(rank + 1) * local]).abs().max().item()
        out_q.put((rank, backend_name, err))
        rm.teardown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not CUDA_OK, reason="needs >= 2 CUDA devices")
@pytest.mark.parametrize("backend", ["ulysses", "ring", "distflash"])
@pytest.mark.parametrize("doc_lengths", PACKS)
def test_varlen_cp_matches_single_gpu(backend, doc_lengths):
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, backend, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=180) for _ in range(world)]
    for p in procs:
        p.join(timeout=180)
    for rank, bk, err in results:
        assert err < 4e-2, f"rank {rank} {bk} varlen vs single-GPU max err {err} (docs={doc_lengths})"
