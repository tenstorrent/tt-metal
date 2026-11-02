# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""w_stream bake-off: the forked op (./fork) with WS_FLAGS (comma list of NAME[=value], "" = baseline) on
WS_SHAPES (comma list of BxS). Correctness-gated with the unit suite's check_outputs. Run via run.sh."""

import hashlib
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


FORK = os.environ.get("WS_FORK", "fork")
fork = _load("w_stream_fork", HERE / FORK / "__init__.py", package=True)
pd = sys.modules["w_stream_fork.qkv_proj_heads_rope_program_descriptor"]
ref = _load("w_stream_ref_test", TESTS / "test_qkv_proj_heads_rope.py")

FLAGS = [f for f in os.environ.get("WS_FLAGS", "").split(",") if f]
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("WS_SHAPES", "2x512").split(",")]


@pytest.fixture
def configured(monkeypatch):
    if hasattr(pd, "WS_FLAGS"):
        monkeypatch.setattr(pd, "WS_FLAGS", FLAGS)


@pytest.mark.parametrize("batch, seq", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
def test_w_stream(device, configured, batch, seq):
    import torch

    x, w, b, cos, sin = ref.make_inputs(batch, seq, seq)
    expected = ref.reference(x, w, b, cos, sin)
    if seq % 32:
        l1 = ttnn.L1_MEMORY_CONFIG
        inf = float("inf")
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
    blocking = pd.select_blocking(device, batch, seq, fork.default_compute_kernel_config())
    stats = []
    for name, out, exp in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        digest = hashlib.sha1(actual.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:10]
        stats.append(f"{name}:sha={digest},pcc={ref.pcc(actual, exp):.6f},rms={ref.relative_rms(actual, exp):.3e}")
    print(
        f"W_STREAM flags={','.join(FLAGS) or 'base'} {batch}x{seq} grid={blocking.grid_m}x{blocking.grid_n} "
        f"block_m={blocking.block_m} wdepth={blocking.weight_depth} " + " ".join(stats)
    )
    ref.check_outputs(outputs, expected, buffer_type=outputs[0].memory_config().buffer_type)
    if seq % 32:
        for name, out in zip("qkv", outputs):
            pad = ref.padded_rows(out, seq)
            assert (pad == 0).all(), f"{name}: {int((pad != 0).sum())} non-zero padded values"
