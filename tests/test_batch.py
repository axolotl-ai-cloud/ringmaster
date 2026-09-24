"""Batch replication preserves metadata inside disjoint CP groups."""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from ringmaster.batch import broadcast_batch
from ringmaster.shard import shard_batch


def _probe():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        group = groups[rank // 2]
        if rank % 2 == 0:
            batch = {"input_ids": torch.full((1, 3 + rank), rank, dtype=torch.long),
                     "labels": torch.ones(1, 3 + rank)}
        else:
            batch = {"extra": torch.zeros(1, 2), "input_ids": torch.zeros(1, 1)}
        broadcast_batch(batch, group)
        assert set(batch) == {"input_ids", "labels"}
        assert batch["input_ids"].shape == (1, 3 + (rank // 2) * 2)
        assert batch["input_ids"].dtype == torch.long
        assert (batch["input_ids"] == (rank // 2) * 2).all()
        print(f"PASS batch rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_subgroup_source_and_heterogeneous_metadata():
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=4", __file__],
        env=os.environ | {"OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS batch rank=") == 4


def test_embedding_shards_pad_sequence_not_hidden_dimension():
    embeddings = torch.arange(30).reshape(1, 5, 6).float().requires_grad_()
    batch, info = shard_batch({"input_ids": embeddings}, cp_rank=1, cp_size=2,
                             load_balance="contiguous")
    assert info.pad_len == 1
    assert batch["input_ids"].shape == (1, 3, 6)
    torch.testing.assert_close(batch["input_ids"][:, :2], embeddings[:, 3:])
    assert not batch["input_ids"][:, 2].any()
    batch["input_ids"].sum().backward()
    assert embeddings.grad[:, 3:].eq(1).all()
    assert not embeddings.grad[:, :3].any()


if __name__ == "__main__":
    _probe()
