# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""[perf experiment dest_resident_k] copy of the op perf harness, driving the experiment copy mrs_drk/.
Perf harness for matmul_reduce_scatter: the feature_spec LOOSE_CASES on the placement their extras pin.

Run under the device profiler:
    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_perf.py
Each case does WARMUP calls (compile + cache) then MEASURED calls; read the per-device
"DEVICE KERNEL DURATION [ns]" of the generic op rows in the profiler CSV. The op's latency is the
MAX over the group's devices of one call (the slowest chip finishes the reduce-scatter).
Correctness is checked on the last call against each case's precision gate.

MRS_PERF_TRACE=1 measures traced: WARMUP eager calls, then one call captured into a trace and replayed MEASURED times
(the like-for-like comparison with the traced unfused baseline; it removes the host dispatch skew between chips).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")  # perf-experiment harness (ttnn forbids a global torch import)
import ttnn

from eval.golden_tests.matmul_reduce_scatter.feature_spec import LOOSE_CASES
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    pytorch_matmul_reduce_scatter,
    quantize_like_device,
    random_stacked,
)
from eval.golden_tests.matmul_reduce_scatter.topology import pinned_mesh_feasible
import copy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrs_drk  # noqa: E402
from mrs_drk import matmul_reduce_scatter  # noqa: E402

# domain-sweep cells beyond LOOSE_CASES: focus shape on Linear; Kimi K3 dense FFN (Kt=264: stream regime)
_lin = copy.deepcopy(LOOSE_CASES[0])
_lin["topology"] = ttnn.Topology.Linear
_k3 = copy.deepcopy(LOOSE_CASES[0])
_k3["inputs"] = ((1, 1, 640, 8448), (8448, 7168))
_k3["extras"] = dict(_k3["extras"])
_k3["extras"].pop("rel_rms_threshold", None)
# single-subblock cell (GLM q_a_proj: P = 4, one 2x2 subblock) and a bf16-W / bf8b-A dtype cell
_narrow = copy.deepcopy(_k3)
_narrow["inputs"] = ((1, 1, 640, 1536), (1536, 2048))
_dt = copy.deepcopy(_k3)
_dt["inputs"] = ((1, 1, 640, 2048), (2048, 6144))
_dt["weight_dtype"] = ttnn.bfloat16
_dt["dtype"] = ttnn.bfloat8_b
CASES = list(LOOSE_CASES) + [_lin, _k3, _narrow, _dt]

WARMUP = 2
MEASURED = 3
TRACE = os.environ.get("MRS_PERF_TRACE", "0") == "1"
TRACE_REGION_BYTES = 1 << 20


def _open(mesh_shape, fabric_config):
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": fabric_config})
    params.pop("fabric_config")
    if TRACE:
        params["trace_region_size"] = TRACE_REGION_BYTES
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)


@pytest.fixture(scope="module")
def perf_mesh():
    ex = LOOSE_CASES[0]["extras"]
    if pinned_mesh_feasible(ex["mesh_shape"], ex["fabric_config"], LOOSE_CASES[0]["topology"], 1) is not None:
        pytest.skip("perf placement not available on this cluster")
    mesh = _open(ex["mesh_shape"], ex["fabric_config"])
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _id(case):
    (a, w) = case["inputs"]
    t = "" if case["topology"] == ttnn.Topology.Ring else "_lin"
    d = "" if case["weight_dtype"] == ttnn.bfloat8_b and case["dtype"] == ttnn.bfloat16 else "_dt"
    return f"M{a[-2]}_K{a[-1]}_N{w[-1]}_sd{case['scatter_dim']}{t}{d}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_perf_case(perf_mesh, case):
    force = os.environ.get("MRS_FORCE")
    mrs_drk._FORCE_XBLOCK_RESIDENCY = None if force is None else bool(int(force))
    ex = case["extras"]
    a_shape, w_shape = case["inputs"]
    a = random_stacked(perf_mesh, a_shape, torch.bfloat16, seed=0)
    w = quantize_like_device(
        random_stacked(perf_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), case["weight_dtype"]
    )
    ta = create_ttnn_input_tensor(a, perf_mesh, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, perf_mesh, dtype=case["weight_dtype"], layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ex["math_fidelity"], fp32_dest_acc_en=ex["fp32_dest_acc_en"])
    run = lambda: matmul_reduce_scatter(
        ta,
        tw,
        cluster_axis=case["cluster_axis"],
        scatter_dim=case["scatter_dim"],
        topology=case["topology"],
        num_links=case["num_links"],
        compute_kernel_config=ckc,
    )
    out = None
    if TRACE:
        for _ in range(WARMUP):
            ttnn.deallocate(run())
        trace_id = ttnn.begin_trace_capture(perf_mesh, cq_id=0)
        out = run()
        ttnn.end_trace_capture(perf_mesh, trace_id, cq_id=0)
        for _ in range(MEASURED):
            ttnn.execute_trace(perf_mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(perf_mesh)
        ttnn.release_trace(perf_mesh, trace_id)
    else:
        for _ in range(WARMUP + MEASURED):
            out = run()
    ttnn.synchronize_device(perf_mesh)
    expected = pytorch_matmul_reduce_scatter(a, w, case["cluster_axis"], case["scatter_dim"])
    cols = tuple(perf_mesh.shape)[1]
    pcc_t, rms_t = TOLERANCES[case["weight_dtype"]]
    rms_t = ex.get("rel_rms_threshold", rms_t)
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).double()
        ref = expected[idx // cols, idx % cols].double()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"DEV {idx} rel_rms {rms:.5f}")
        if not os.environ.get("MRS_ABLATE"):
            assert rms <= rms_t, f"device {idx}: rel_rms {rms} > {rms_t}"
    pkg = mrs_drk

    print(f"PLAN {_id(case)}: {pkg._LAST_PLAN} rel_rms(last device) {rms:.5f}")
