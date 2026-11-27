# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tx_early_start: per-call wall time under trace replay (same method as the op's
test_a2a_wkv_b2_trace_timing.py), variants alternated in one session: order from TXE_ORDER
(comma list of variant names, default base,early,early,base)."""

import os
import sys
import time

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import txe_descriptor  # noqa: E402
from txe_op import a2a_wkv_b2  # noqa: E402
from test_txe_accept import MESH_SHAPE, _make_inputs  # noqa: E402

N_REPLAYS = 50
ORDER = [v for v in os.environ.get("TXE_ORDER", "base,early,early,base").split(",") if v]


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("slot", list(range(len(ORDER))))
def test_txe_trace_timing(mesh_device, slot):
    variant = ORDER[slot]
    txe_descriptor.VARIANT = variant
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2(attn, w))  # compile + semaphore creation (eager)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(N_REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REPLAYS * 1e6
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    print(f"\nTRACE_TIMING txe[{variant}]: {us:.1f} us per call (mean of {N_REPLAYS} replays)")
