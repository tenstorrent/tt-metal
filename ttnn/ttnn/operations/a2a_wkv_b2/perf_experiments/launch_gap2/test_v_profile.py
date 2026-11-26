"""launch_gap2: device profile of variant V_VARIANTS[0]: R back-to-back replays of a one-call trace. Run with
TT_METAL_DEVICE_PROFILER=1; parse with ../launch_gap/lg_gaps.py <csv> 1 R (or perf/trace_gaps.py)."""
import os

import pytest
import ttnn

from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap2.v_op import v_op, apply
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs

R = int(os.environ.get("V_R", "6"))


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 22}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_v_profile(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    apply(os.environ.get("V_VARIANTS", "base").split(",")[0])
    ttnn.deallocate(v_op(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = v_op(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(R):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    ttnn.ReadDeviceProfiler(mesh_device)
    apply("base")
