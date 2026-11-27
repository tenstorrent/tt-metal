"""Device profile of back-to-back trace replays (Refinement 3): per-replay device span and the gap between
consecutive replays on each chip. Parse with perf/trace_gaps.py."""
import pytest, ttnn
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ro_op import a2a_wkv_b2  # noqa: E402
from test_ro_accept import MESH_SHAPE, _make_inputs


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_trace_profile(mesh_device):
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
