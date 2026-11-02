# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""m_balance bake-off: the forked op (./fork) with MB_PLAN (base | A_mc | A_uc | B; forced only at Mt = 64) on
MB_SHAPES (comma list of BxS). Correctness-gated with the unit suite's check_outputs (PCC >= 0.9999,
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


fork = _load("m_balance_fork", HERE / "fork/__init__.py", package=True)
pd = sys.modules["m_balance_fork.qkv_proj_heads_rope_program_descriptor"]
ref = _load("m_balance_ref_test", TESTS / "test_qkv_proj_heads_rope.py")

SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("MB_SHAPES", "8x512").split(",")]


@pytest.mark.parametrize("batch, seq", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
def test_m_balance(device, batch, seq):
    x, w, b, cos, sin = ref.make_inputs(batch, seq, seq)
    expected = ref.reference(x, w, b, cos, sin)
    if seq % 32:
        inf = float("inf")
        l1 = ttnn.L1_MEMORY_CONFIG
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
    cfg = fork.default_compute_kernel_config()
    blocking = pd.select_blocking(device, batch, seq, cfg)
    regions = pd.plan_regions(device, blocking, cfg)
    plan = ";".join(
        f"h{r.blocking.heads_per_core}:{r.blocking.grid_m}x{r.blocking.grid_n}:bm{r.blocking.block_m}:"
        f"{'mc' if r.w_multicast else 'uc'}"
        for r in regions
    )
    stats = []
    for name, out, exp in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        stats.append(f"{name}:pcc={ref.pcc(actual, exp):.6f},rms={ref.relative_rms(actual, exp):.3e}")
    print(f"M_BALANCE plan={pd.MB_PLAN}{'+x' if pd.MB_XSPLIT else ''} {batch}x{seq} regions={plan} " + " ".join(stats))
    if os.environ.get("QKV_ABLATE"):
        return  # ablated kernels compute garbage by design
    ref.check_outputs(outputs, expected, buffer_type=outputs[0].memory_config().buffer_type)
    if seq % 32:
        for name, out in zip("qkv", outputs):
            pad = ref.padded_rows(out, seq)
            assert (pad == 0).all(), f"{name}: {int((pad != 0).sum())} non-zero padded values"
