# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""launch_gap: device profile of one variant (LG_VARIANTS[0]): eager call, then LG_R replays of a trace of
LG_K calls. Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir>; parse with lg_gaps.py."""
import os

import pytest
import ttnn

from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap.test_lg_timing import apply, NAMES, K
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap.lg_op import lg_op
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs

R = int(os.environ.get("LG_R", "6"))


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 22}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_lg_profile(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    pad = None
    if os.environ.get("LG_PAD", "1") == "1":  # a real-op trace replayed first so the host is ahead of the device
        apply("real")
        ttnn.deallocate(lg_op(attn, w))
        pad = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        pad_out = lg_op(attn, w)
        ttnn.end_trace_capture(mesh_device, pad, cq_id=0)
        ttnn.synchronize_device(mesh_device)
    apply(NAMES[0])
    ttnn.deallocate(lg_op(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [lg_op(attn, w) for _ in range(K)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    if pad is not None:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, pad, cq_id=0, blocking=False)
    for _ in range(R):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    if pad is not None:
        ttnn.release_trace(mesh_device, pad)
    ttnn.ReadDeviceProfiler(mesh_device)
    apply("real")
