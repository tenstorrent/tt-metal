# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""bias_seed bake-off: the forked qkv_proj_heads_rope (./fork) with BS_MODE / BS_KB selecting how the bias
enters the Float32 accumulator. Run with scripts/run_safe_pytest.sh --profile; parse with parse.py.

BS_MODE: ktile (baseline) | ktile_raw | partials_seed | dest_seed.  BS_KB: k_block for the K' = 24 modes.
BS_SHAPES: comma list of BxS.  BS_HEADS: force heads_per_core (0 = selection's pick)."""

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


fork = _load("bias_seed_fork", HERE / "fork/__init__.py", package=True)
pd = sys.modules["bias_seed_fork.qkv_proj_heads_rope_program_descriptor"]
ref = _load("bias_seed_ref_test", TESTS / "test_qkv_proj_heads_rope.py")

MODE = os.environ.get("BS_MODE", "ktile")
KB = int(os.environ.get("BS_KB", "4"))
HEADS = int(os.environ.get("BS_HEADS", "0"))
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("BS_SHAPES", "8x512").split(",")]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(pd, "BIAS_MODE", MODE)
    monkeypatch.setattr(pd, "K_BLOCK_SEEDED", KB)
    if HEADS:
        monkeypatch.setattr(pd, "HEADS_PER_CORE_CANDIDATES", (HEADS,))
    if "BS_BUDGET_KB" in os.environ:
        monkeypatch.setattr(pd, "CB_BUDGET_BYTES", int(os.environ["BS_BUDGET_KB"]) * 1024)


def _report(batch, seq, outputs, expected, device):
    import torch

    blocking = pd.select_blocking(device, batch, seq, fork.default_compute_kernel_config())
    w_mc = pd._w_multicast(
        device.compute_with_storage_grid_size(),
        blocking.grid_m,
        blocking.grid_n,
        blocking.tensor_row_tiles,
        blocking.block_m,
    )
    stats = []
    for name, out, exp in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        stats.append(f"{name}:pcc={ref.pcc(actual, exp):.6f},rms={ref.relative_rms(actual, exp):.3e}")
    print(
        f"BIAS_SEED {MODE} kb={blocking.k_block} {batch}x{seq} grid={blocking.grid_m}x{blocking.grid_n} "
        f"block_m={blocking.block_m} sb={blocking.sb_h}x{blocking.sb_w} wdepth={blocking.weight_depth} "
        f"r2={w_mc} " + " ".join(stats)
    )


@pytest.mark.parametrize("batch, seq", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
def test_bias_seed(device, configured, batch, seq):
    x, w, b, cos, sin = ref.make_inputs(batch, seq, seq)
    expected = ref.reference(x, w, b, cos, sin)
    if seq % 32:
        # Non-aligned S: inf in the x / table tile padding; output padding rows must be exact zero.
        inf = float("inf")
        l1 = ttnn.L1_MEMORY_CONFIG
        # B=4096, S=1 tile-pads x and each output to ~100-200 MB: DRAM only (as the unit suite).
        big = ttnn.DRAM_MEMORY_CONFIG if batch * ref.ceil_tile(seq) > 8192 else l1
        tensors = (
            ref.to_device_padded(x, device, big, inf),
            ref.to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
            ref.to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
            ref.to_device_padded(cos, device, l1, inf),
            ref.to_device_padded(sin, device, l1, inf),
        )
    else:
        tensors = ref.device_inputs(device, x, w, b, cos, sin)
    out_memory = big if seq % 32 else None
    outputs = fork.qkv_proj_heads_rope(*tensors, memory_config=out_memory)
    _report(batch, seq, outputs, expected, device)
    ref.check_outputs(outputs, expected, buffer_type=outputs[0].memory_config().buffer_type)
    if seq % 32:
        for name, out in zip("qkv", outputs):
            pad = ref.padded_rows(out, seq)
            assert (pad == 0).all(), f"{name}: {int((pad != 0).sum())} non-zero padded values"
