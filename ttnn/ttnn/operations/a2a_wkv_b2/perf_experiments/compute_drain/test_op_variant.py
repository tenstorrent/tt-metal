# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_drain: the whole op with its worker compute swapped (KERNEL_DIR patched to a copy of the op's
kernels; the real op is untouched). Env CD_OPS = comma list of op_kernels_<name> dirs, run in that order.
  test_op_correct     : every variant vs fp64 (op tolerance) and bit-identity vs the first variant
  test_op_trace_timing: trace-replay us per call per variant (the op's trace timing test)
  test_op_profile     : 6 trace replays + ReadDeviceProfiler (parse with drain_from_profile.py)"""
import os
import time
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.a2a_wkv_b2.a2a_wkv_b2_program_descriptor as pdmod
from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import (
    MESH_SHAPE,
    _make_inputs,
    _expected,
    _check,
)

HERE = Path(__file__).parent
OPS = [v for v in os.environ.get("CD_OPS", "base,chunkpush").split(",") if v]
_trace = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
_mesh = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)


def _use(name):
    pdmod.KERNEL_DIR = HERE / f"op_kernels_{name}"


@_trace
@_mesh
@pytest.mark.parametrize("seed,scale", [(0, 1.0), (3, 8.0)])
def test_op_correct(mesh_device, seed, scale):
    import torch

    chips, attn, w = _make_inputs(mesh_device, seed, scale=scale)
    exp = _expected(mesh_device, chips, w)
    ref = None
    for name in OPS:
        _use(name)
        out = _check(a2a_wkv_b2(attn, w), exp, mesh_device)
        cur = torch.stack(out)
        if ref is None:
            ref = cur
        print(f"\nCD_OP_CORRECT {name} seed={seed} scale={scale} bit-identical-to-{OPS[0]}={torch.equal(ref, cur)}")


@_trace
@_mesh
def test_op_trace_timing(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    for name in OPS:
        _use(name)
        ttnn.deallocate(a2a_wkv_b2(attn, w))
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = a2a_wkv_b2(attn, w)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(50):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / 50 * 1e6
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(out)
        print(f"\nTRACE_TIMING {name}: {us:.1f} us per call (mean of 50 replays)")


@_trace
@_mesh
def test_op_profile(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    _use(OPS[0])
    ttnn.deallocate(a2a_wkv_b2(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(6):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    ttnn.ReadDeviceProfiler(mesh_device)
