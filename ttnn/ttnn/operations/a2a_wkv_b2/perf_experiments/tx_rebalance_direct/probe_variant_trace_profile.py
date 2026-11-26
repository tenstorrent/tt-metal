"""Device profile of 6 back-to-back trace replays of the variant. Parse with parse_links.py."""
import os
import sys

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variant_op import a2a_wkv_b2_variant  # noqa: E402
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs  # noqa: E402


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_variant_trace_profile(mesh_device):
    _, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2_variant(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2_variant(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(6):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    ttnn.ReadDeviceProfiler(mesh_device)
