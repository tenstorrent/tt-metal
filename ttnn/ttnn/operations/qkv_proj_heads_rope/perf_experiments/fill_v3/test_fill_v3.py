# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fill_v3 bake-off: the forked op (./fork) with F3_FLAGS (comma list of kernel defines, "" = baseline) on
F3_SHAPES (comma list of BxS). Inputs placed as in the perf guard (4096x1 all DRAM, else x / tables / outputs
in L1; W / bias always DRAM). Correctness-gated with the unit suite's check_outputs (PCC >= 0.9999,
rel RMS <= 4e-3). Run via run.sh (profiled)."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
TESTS = Path(__file__).resolve().parents[6] / "tests/ttnn/unit_tests/operations/qkv_proj_heads_rope"


def _load(name, path, package=False):
    kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fork = _load("fill_v3_fork", HERE / "fork/__init__.py", package=True)
pd = sys.modules["fill_v3_fork.qkv_proj_heads_rope_program_descriptor"]
ref = _load("fill_v3_ref_test", TESTS / "test_qkv_proj_heads_rope.py")

FLAGS = [f for f in os.environ.get("F3_FLAGS", "").split(",") if f]
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("F3_SHAPES", "4x512").split(",")]
DUMP = os.environ.get("F3_DUMP")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(pd, "F3_FLAGS", FLAGS)


@pytest.mark.parametrize("batch, seq", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
def test_fill_v3(device, configured, batch, seq):
    import torch

    x, w, b, cos, sin = ref.make_inputs(batch, seq, seq)
    expected = ref.reference(x, w, b, cos, sin)
    mc = ttnn.DRAM_MEMORY_CONFIG if batch * ref.ceil_tile(seq) > 8192 else ttnn.L1_MEMORY_CONFIG
    tensors = (
        ref.to_device(x, device, mc),
        ref.to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
        ref.to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
        ref.to_device(cos, device, mc),
        ref.to_device(sin, device, mc),
    )
    outputs = fork.qkv_proj_heads_rope(*tensors, memory_config=mc)
    blocking = pd.select_blocking(device, batch, seq, fork.default_compute_kernel_config())
    stats = []
    actuals = []
    for name, out, exp in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        actuals.append(actual)
        stats.append(f"{name}:pcc={ref.pcc(actual, exp):.7f},rms={ref.relative_rms(actual, exp):.3e}")
    if DUMP:
        torch.save(actuals, f"{DUMP}_{batch}x{seq}.pt")
    print(
        f"FILL_V3 flags={','.join(FLAGS) or 'base'} {batch}x{seq} grid={blocking.grid_m}x{blocking.grid_n} "
        f"block_m={blocking.block_m} wdepth={blocking.weight_depth} " + " ".join(stats)
    )
    if any(f.startswith("F3_ABL_") for f in FLAGS):
        return  # ablation: timing only
    ref.check_outputs(outputs, expected, buffer_type=outputs[0].memory_config().buffer_type)
