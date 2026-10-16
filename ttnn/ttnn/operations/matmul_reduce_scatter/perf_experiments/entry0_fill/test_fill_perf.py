# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""entry0_fill bake-off: the whole op on the LOOSE_CASES (+ a Linear cell), with the reader variant chosen by env.

The op runs through an experiment copy of its descriptor (mrs_fill_descriptor.py, kernels/ next to it), installed in
place of the real planner module, so the real op files are untouched. Variant = MRS_FILL="DEFINE[=V],..." (reader
defines). Measure traced:
    MRS_PERF_TRACE=1 MRS_FILL=... scripts/run_safe_pytest.sh --profile <this file> -k <case>
Correctness gate: every device's rel_rms vs torch within the case's threshold.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

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

_REAL = "ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor"
_spec = importlib.util.spec_from_file_location(_REAL, Path(__file__).parent / "mrs_fill_descriptor.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[_REAL] = _mod  # the op imports its planner lazily (`from .matmul_reduce_scatter_program_descriptor ...`)
import ttnn.operations.matmul_reduce_scatter as _pkg  # noqa: E402

_pkg.matmul_reduce_scatter_program_descriptor = _mod
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter  # noqa: E402

WARMUP = 2
MEASURED = int(os.environ.get("MRS_MEASURED", 3))
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
    topo = "_lin" if case["topology"] == ttnn.Topology.Linear else ""
    return f"M{a[-2]}_K{a[-1]}_N{w[-1]}_sd{case['scatter_dim']}{topo}"


CASES = list(LOOSE_CASES) + [dict(LOOSE_CASES[0], topology=ttnn.Topology.Linear)]


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_fill_case(perf_mesh, case):
    ex = case["extras"]
    a_shape, w_shape = case["inputs"]
    a = random_stacked(perf_mesh, a_shape, torch.bfloat16, seed=0)
    w = quantize_like_device(
        random_stacked(perf_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), case["weight_dtype"]
    )
    ta = create_ttnn_input_tensor(a, perf_mesh, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, perf_mesh, dtype=case["weight_dtype"], layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ex["math_fidelity"], fp32_dest_acc_en=ex["fp32_dest_acc_en"])
    if os.environ.get("MRS_FORCE_STREAM") == "1":  # domain sweep: force the stream (non-resident) regime
        _pkg._FORCE_XBLOCK_RESIDENCY = False
    run = lambda: matmul_reduce_scatter(
        ta,
        tw,
        cluster_axis=case["cluster_axis"],
        scatter_dim=case["scatter_dim"],
        topology=case["topology"],
        num_links=int(os.environ.get("MRS_LINKS", case["num_links"])),
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
    worst = 0.0
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).double()
        ref = expected[idx // cols, idx % cols].double()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        worst = max(worst, rms)
        assert rms <= rms_t, f"device {idx}: rel_rms {rms} > {rms_t}"
    print(
        f"FILL_RESULT {_id(case)} variant='{os.environ.get('MRS_FILL', '')}' worst_rel_rms {worst:.5f} plan {_pkg._LAST_PLAN}"
    )


# ---- correctness sweep (eager): shapes / dtypes / regimes beyond the perf cases ----
_SWEEP = [
    # (A, W, scatter_dim, dtype, weight_dtype, topology, num_links, force_stream)
    ((1, 1, 640, 512), (512, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Ring, 2, False),
    ((1, 1, 640, 1536), (1536, 128), -1, ttnn.bfloat16, ttnn.bfloat16, ttnn.Topology.Ring, 2, False),
    ((1, 1, 640, 1536), (1536, 32), -2, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Ring, 1, False),
    ((1, 1, 2048, 4096), (4096, 4096), -2, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Ring, 2, False),
    ((640, 2048), (2048, 7168), -1, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.Topology.Linear, 2, False),
    ((1, 1, 640, 3072), (3072, 6144), -1, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Linear, 1, False),
    ((1, 1, 640, 2048), (2048, 7168), -2, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Ring, 2, True),
    ((1, 1, 640, 1792), (1792, 1536), -1, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.Topology.Ring, 2, False),
]


@pytest.mark.parametrize("cfg", _SWEEP, ids=[f"sw{i}" for i in range(len(_SWEEP))])
def test_fill_correctness(perf_mesh, cfg):
    a_shape, w_shape, sd, dt, wdt, topo, links, force_stream = cfg
    a = random_stacked(perf_mesh, a_shape, torch.bfloat16, seed=0)
    w = quantize_like_device(random_stacked(perf_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), wdt)
    ta = create_ttnn_input_tensor(a, perf_mesh, dtype=dt, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, perf_mesh, dtype=wdt, layout=ttnn.TILE_LAYOUT)
    _pkg._FORCE_XBLOCK_RESIDENCY = False if force_stream else None
    try:
        for _ in range(2):  # second call reuses the program cache (counter re-arm path)
            out = matmul_reduce_scatter(ta, tw, cluster_axis=1, scatter_dim=sd, topology=topo, num_links=links)
    finally:
        _pkg._FORCE_XBLOCK_RESIDENCY = None
    ttnn.synchronize_device(perf_mesh)
    expected = pytorch_matmul_reduce_scatter(a, w, 1, sd)
    cols = tuple(perf_mesh.shape)[1]
    _, rms_t = TOLERANCES[wdt]
    worst = 0.0
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).double()
        ref = expected[idx // cols, idx % cols].double()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        worst = max(worst, rms)
        assert rms <= rms_t, f"device {idx}: rel_rms {rms} > {rms_t}"
    print(
        f"SWEEP_RESULT {cfg[:3]} kdir={os.environ.get('MRS_FILL_KDIR', 'kernels')} worst_rel_rms {worst:.5f} plan {_pkg._LAST_PLAN}"
    )
