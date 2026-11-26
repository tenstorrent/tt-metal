# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-call trace-replay wall time of the tx_rebalance_direct variant (copy of the op's
test_a2a_wkv_b2_trace_timing.py, pointed at variant_op)."""
import os
import sys
import time

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variant_op import a2a_wkv_b2_variant, config_tag  # noqa: E402
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs  # noqa: E402

N_REPLAYS = 50


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_variant_trace_timing(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2_variant(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2_variant(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(N_REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REPLAYS * 1e6
    ttnn.release_trace(mesh_device, tid)
    print(f"\nTRACE_TIMING variant [{config_tag()}]: {us:.1f} us per call (mean of {N_REPLAYS} replays)")
