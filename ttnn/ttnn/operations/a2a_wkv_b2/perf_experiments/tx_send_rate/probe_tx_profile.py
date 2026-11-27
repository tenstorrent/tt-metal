# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device profile of 6 back-to-back trace replays of the tx_send_rate variant in env TXV (parse with
the op's perf/trace_gaps.py, perf/stage_zones.py, and this dir's txd.py)."""
import os, sys
import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tx_op import a2a_wkv_b2
from test_tx_accept import MESH_SHAPE, _make_inputs


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_tx_profile(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(6):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    ttnn.ReadDeviceProfiler(mesh_device)
