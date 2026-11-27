"""launch_gap2: per-call wall time (one call per trace, 50 replays) of variants V_VARIANTS (comma list, run in
that order in one process; alternate the order between runs). Also checks each variant's output bit-matches
the 'base' variant's output (one eager call each)."""
import os
import time

import pytest
import ttnn

from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap2.v_op import v_op, apply
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs

NAMES = [v for v in os.environ.get("V_VARIANTS", "base,merge").split(",") if v]
N_REPLAYS = int(os.environ.get("V_N", "50"))


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 22}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_v_timing(mesh_device):
    import torch

    _, attn, w = _make_inputs(mesh_device, 0)
    ref, res = None, []
    for name in NAMES:
        apply(name, info=os.environ.get("V_INFO") == "1")
        mesh_device.clear_program_cache()
        o = v_op(attn, w)
        t = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
        ttnn.deallocate(o)
        if ref is None:
            ref = t
        same = torch.equal(t, ref)
        apply(name)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = v_op(attn, w)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(N_REPLAYS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / N_REPLAYS * 1e6
        t2 = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
        same = same and torch.equal(t2, ref)
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(out)
        res.append((name, us, same))
        print(f"\nV_TIMING {name}: {us:.2f} us per call, output==first variant: {same}", flush=True)
    apply("base")
    print("\nV_SUMMARY " + " | ".join(f"{n}={u:.2f}{'' if s else '(MISMATCH)'}" for n, u, s in res))
    assert all(s for _, _, s in res), "variant output differs from the first variant's"
