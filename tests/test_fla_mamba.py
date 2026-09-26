"""Optional FLA discovery and independent CPU recurrence controls."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ringmaster.fla_mamba import fla_mamba_mixers


def test_fla_discovery_does_not_require_fla_import():
    mixer_class = type("Mamba", (torch.nn.Module,), {"__module__": "fla.layers.mamba"})
    mixer = mixer_class()
    model = torch.nn.Sequential(mixer, torch.nn.Linear(3, 3))
    assert fla_mamba_mixers(model) == [mixer]


@pytest.mark.slow
def test_mamba1_cpu_dp2_cp4_output_and_gradient_parity():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            str(Path(__file__).with_name("_fla_mamba_scan1_probe.py")),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == 16


def test_missing_scan_restores_previously_wired_mixers(monkeypatch):
    from types import FunctionType, MethodType

    from ringmaster.fla_mamba import wire_fla_mamba

    class Mixer(torch.nn.Module):
        def __init__(self, scan):
            super().__init__()
            self.backend = "cuda"
            self.causal_conv1d_fn = lambda *args, **kwargs: None
            raw = type(self).cuda_kernels_forward
            self.cuda_kernels_forward = MethodType(
                FunctionType(
                    raw.__code__,
                    {**raw.__globals__, "selective_scan_fn": scan},
                    raw.__name__,
                ),
                self,
            )

        def cuda_kernels_forward(self, value):
            return value

        def forward(self, value):
            return value

    first, second = Mixer(lambda *args, **kwargs: None), Mixer(None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    with pytest.raises(ValueError, match="loaded Mamba scan kernels"):
        wire_fla_mamba([first, second], object())
    assert "forward" not in vars(first)
    assert "forward" not in vars(second)
