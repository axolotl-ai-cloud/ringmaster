"""Two-rank model parity for sliding, chunked, and packed attention."""

import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest


def test_balanced_query_temperature_uses_global_positions(monkeypatch):
    from ringmaster.strategies import query_scaling

    runtime = SimpleNamespace(shard_load_balance="head_tail", cp_rank=1, cp_size=2)
    monkeypatch.setattr(query_scaling, "get_runtime", lambda: runtime)
    module = SimpleNamespace(
        attn_temperature_tuning=True, use_rope=False, floor_scale=4, attn_scale=1
    )
    query = torch.ones(1, 1, 8, 1)
    actual = query_scaling.global_query_scale(module, query).flatten()
    local = torch.arange(8)
    global_positions = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])

    def scale(positions):
        return 1 + torch.log1p(torch.floor((positions + 1) / 4))

    torch.testing.assert_close(actual, scale(global_positions) / scale(local))


def _model(family):
    if family == "sliding":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            use_sliding_window=True,
            sliding_window=4,
            max_window_layers=0,
            attn_implementation="sdpa",
            use_cache=False,
        )
        return Qwen3ForCausalLM(config)

    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

    config = Llama4TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        intermediate_size_mlp=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        moe_layers=[],
        no_rope_layers=[1, 0],
        layer_types=["chunked_attention", "full_attention"],
        attention_chunk_size=4,
        attn_temperature_tuning=True,
        floor_scale=4,
        attn_implementation="sdpa",
        use_cache=False,
    )
    return Llama4ForCausalLM(config)


def _worker(rank, queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29637")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        import ringmaster as rm
        from ringmaster.shard import shard_batch, varlen_meta
        from ringmaster.strategies.ring import make_ring_attention
        from transformers import AttentionInterface

        results = []
        for family in ("sliding", "chunked"):
            for packed in (False, True):
                for backend in (rm.Backend.ULYSSES, rm.Backend.RING):
                    torch.manual_seed(0)
                    model = _model(family).eval()
                    reference = copy.deepcopy(model)
                    torch.manual_seed(42)
                    ids = torch.randint(3, 64, (1, 16))
                    positions = (
                        torch.cat((torch.arange(7), torch.arange(9))).unsqueeze(0)
                        if packed
                        else torch.arange(16).unsqueeze(0)
                    )
                    expected = reference(
                        input_ids=ids, position_ids=positions, use_cache=False
                    ).logits
                    expected.sum().backward()
                    expected_grads = {
                        name: param.grad.clone()
                        for name, param in reference.named_parameters()
                        if param.grad is not None
                    }

                    runtime = rm.setup(
                        rm.RingmasterConfig(size=2, backend=backend),
                        num_kv_heads=4,
                        device_type="cpu",
                        inner_attn="sdpa",
                    )
                    if backend == rm.Backend.RING:
                        AttentionInterface.register(
                            runtime.attn_implementation,
                            make_ring_attention(
                                "math", "math", rm.RotateMethod.ALLGATHER
                            ),
                        )
                    model.set_attn_implementation(runtime.attn_implementation)
                    runtime.varlen = varlen_meta(positions, 16)
                    batch, _ = shard_batch(
                        {"input_ids": ids.clone(), "position_ids": positions.clone()},
                        cp_rank=rank,
                        cp_size=2,
                    )
                    actual = model(**batch, use_cache=False).logits
                    actual.sum().backward()
                    max_grad_error = 0.0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad)
                            max_grad_error = max(
                                max_grad_error,
                                (param.grad - expected_grads[name]).abs().max().item(),
                            )
                    max_output_error = (
                        (actual - expected[:, rank * 8 : (rank + 1) * 8])
                        .abs()
                        .max()
                        .item()
                    )
                    results.append(
                        (
                            family,
                            packed,
                            backend.value,
                            max_output_error,
                            max_grad_error,
                        )
                    )
                    rm.teardown()
        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_sliding_and_chunked_model_parity():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(rank, queue)) for rank in range(2)]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=180) for _ in workers]
    for worker in workers:
        worker.join(timeout=180)
        assert worker.exitcode == 0
    for rank, cases in results:
        for family, packed, backend, output_error, grad_error in cases:
            assert output_error < 1e-5, (rank, family, packed, backend, output_error)
            assert grad_error < 1e-4, (rank, family, packed, backend, grad_error)
