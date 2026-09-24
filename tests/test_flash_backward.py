"""Opt-in CUDA accuracy and gradient checks for the dense Ring schedules."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_flash_ring_backward_gqa():
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(Path(__file__).with_name("_flash_backward_probe.py"))],
        env=os.environ | {"OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS ") == 6
