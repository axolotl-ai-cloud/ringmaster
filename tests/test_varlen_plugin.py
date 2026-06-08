"""Varlen through the axolotl plugin's data path: ContextParallelContextManager.

The plugin installs `ContextParallelContextManager` (broadcast -> shard -> stash
cu_seqlens -> attention). This drives that exact manager with a PACKED batch and
checks each rank's shard logits equal the single-GPU flash-varlen reference, for
every backend/layout the plugin can build (ulysses, ring, zigzag, distflash).
Requires >= 2 CUDA devices.
"""

import os

import pytest
import torch

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() >= 2

# (backend, load_balance)
CONFIGS = [
    ("ulysses", "none"),
    ("ulysses", "head_tail"),  # guard: Ulysses must shard contiguous despite head_tail
    ("ring", "none"),
    ("ring", "head_tail"),     # genuine zigzag varlen via the plugin
    ("ring", "distflash"),     # genuine distflash varlen via the plugin
]


def _worker(rank, world, backend_name, lb_name, doc_lengths, out_q):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29616")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster import ContextParallelContextManager
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
        pos = torch.cat([torch.arange(n, device=rank) for n in doc_lengths]).unsqueeze(0)

        with torch.no_grad():
            ref = model(input_ids=input_ids, position_ids=pos).logits.float()

        backend = rm.Backend.ULYSSES if backend_name == "ulysses" else rm.Backend.RING
        runtime = rm.setup(
            rm.RingmasterConfig(size=world, backend=backend,
                                load_balance=rm.LoadBalance(lb_name)),
            num_kv_heads=cfg.num_key_value_heads, device_mesh=None,
            inner_attn="flash_attention_2",
        )
        model.set_attn_implementation(runtime.attn_implementation)

        # install exactly as the plugin does (load_balance = shard_load_balance guard)
        actual_lb = runtime.shard_load_balance
        cm = ContextParallelContextManager(
            [model], runtime.cp_group, gather_outputs=False, load_balance=actual_lb,
        )
        cm.install()
        with torch.no_grad():
            out = model(input_ids=input_ids.clone(), position_ids=pos.clone()).logits.float()
        cm.remove()

        # this rank's shard positions follow the ACTUAL layout the guard chose
        if actual_lb == "head_tail":
            from ringmaster.ring.zigzag import _zigzag_gidx
            half = seq // (2 * world)
            idx = _zigzag_gidx(rank, half, world, ref.device)
        else:
            L = seq // world
            idx = torch.arange(rank * L, (rank + 1) * L, device=ref.device)
        err = (out - ref[:, idx]).abs().max().item()
        out_q.put((rank, backend_name, lb_name, err))
        rm.teardown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not CUDA_OK, reason="needs >= 2 CUDA devices")
@pytest.mark.parametrize("backend,lb", CONFIGS)
@pytest.mark.parametrize("doc_lengths", [[24, 40], [16, 16, 32]])
def test_plugin_contextmanager_varlen(backend, lb, doc_lengths):
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, backend, lb, doc_lengths, out_q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=180) for _ in range(world)]
    for p in procs:
        p.join(timeout=180)
    for rank, bk, lbn, err in results:
        assert err < 4e-2, f"rank {rank} plugin varlen {bk}/{lbn} err {err} (docs={doc_lengths})"
