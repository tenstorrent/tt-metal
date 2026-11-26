# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-call wall time under trace replay for the tx_send_pipeline variant in env TXP (copy of the op's
test_a2a_wkv_b2_trace_timing.py pointed at tx_op)."""
import os, sys, time
import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tx_op import a2a_wkv_b2
from test_tx_accept import MESH_SHAPE, _make_inputs

N_REPLAYS = 50


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_tx_timing(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(N_REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REPLAYS * 1e6
    ttnn.release_trace(mesh_device, tid)
    print(
        f"\nTRACE_TIMING tx_send_pipeline[{os.environ.get('TXP','base')}|{os.environ.get('A2A_ABLATE','')}]: "
        f"{us:.1f} us per call (mean of {N_REPLAYS} replays)"
    )
