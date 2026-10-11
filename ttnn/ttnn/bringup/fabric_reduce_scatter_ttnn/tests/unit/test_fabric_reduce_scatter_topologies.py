# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter over every group the box can form, as fabric_all_gather's tests: a mesh axis or the whole mesh
(cluster_axis None: a snake), as a line or a ring, and on a torus with both sides >= 3 the two Hamiltonian cycles.
Each chip's output is checked against the torch sum of its block over its group (block = the chip's row-major rank in
the group), and the wall time per call is printed. A group the fabric cannot route (no direct link for a hop) is
reported as unsupported and skipped (RS_STRICT=1 fails instead).

  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_reduce_scatter_topologies.py -s -rs

RS_MESH=RxC (default 2x4, FABRIC_2D), RS_FABRIC (a FabricConfig name), RS_ROWS (rows per chip of the partial, default
4096), RS_LINKS (default 2), RS_CALLS (timed calls, default 10). An 8-chip ring along axis 0 on a LoudBox:
RS_MESH=8x1 RS_FABRIC=FABRIC_2D_TORUS_Y TT_MESH_GRAPH_DESC_PATH=<an 8x1 [RING] descriptor>."""

import os
import time

import pytest
import torch

import ttnn

MESH = tuple(int(v) for v in os.environ.get("RS_MESH", "2x4").split("x"))
FABRIC = getattr(ttnn.FabricConfig, os.environ.get("RS_FABRIC", "FABRIC_2D"))
ROWS = int(os.environ.get("RS_ROWS", "4096"))
LINKS = int(os.environ.get("RS_LINKS", "2"))
CALLS = int(os.environ.get("RS_CALLS", "10"))
H = 4096

CASES = [
    pytest.param(0, ttnn.Topology.Linear, "ring", id="axis0_line"),
    pytest.param(1, ttnn.Topology.Linear, "ring", id="axis1_line"),
    pytest.param(0, ttnn.Topology.Ring, "ring", id="axis0_ring"),
    pytest.param(1, ttnn.Topology.Ring, "ring", id="axis1_ring"),
    pytest.param(None, ttnn.Topology.Linear, "ring", id="mesh_snake_line"),
    pytest.param(None, ttnn.Topology.Ring, "ring", id="mesh_snake_ring"),
    pytest.param(None, ttnn.Topology.Ring, "dual_cycles", id="mesh_dual_cycles"),
]


def _group_and_slot(r, c, rows, cols, cluster_axis):
    """(the chips of (r, c)'s group as a list of coords, its output slot = row-major rank in the group)."""
    if cluster_axis == 0:
        return [(i, c) for i in range(rows)], r
    if cluster_axis == 1:
        return [(r, i) for i in range(cols)], c
    return [(i, j) for i in range(rows) for j in range(cols)], r * cols + c


@pytest.mark.parametrize("device_params", [{"fabric_config": FABRIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True, ids=[f"{MESH[0]}x{MESH[1]}"])
@pytest.mark.parametrize("cluster_axis, topology, scheme", CASES)
def test_fabric_rs_topology(mesh_device, cluster_axis, topology, scheme):
    from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

    rows, cols = tuple(mesh_device.shape)
    G = rows * cols if cluster_axis is None else (rows, cols)[cluster_axis]
    if G < 2:
        pytest.skip("one chip in the group")
    if scheme == "dual_cycles" and (rows < 3 or cols < 3):
        pytest.skip("dual_cycles needs a torus with both sides >= 3")
    T = ROWS - ROWS % (32 * G)
    torch.manual_seed(7)
    parts = torch.randn(rows, cols, T, H).bfloat16()
    x = ttnn.from_torch(
        parts,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x = ttnn.reshape(x, (1, 1, T, H))
    call = lambda: fabric_reduce_scatter(  # noqa: E731
        x, cluster_axis=cluster_axis, topology=topology, num_links=LINKS, scheme=scheme
    )
    try:
        out = call()
    except ValueError as e:  # no direct link for a hop, or too few links
        if os.environ.get("RS_STRICT") == "1":
            raise
        pytest.skip(f"unsupported on this fabric: {e}")
    Sb = T // G
    worst, rel = 0.0, 0.0
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(d, cols)
        grp, slot = _group_and_slot(r, c, rows, cols, cluster_axis)
        want = sum(parts[i, j, slot * Sb : (slot + 1) * Sb].float() for i, j in grp)
        got = ttnn.to_torch(t).float().reshape(Sb, H)
        worst = max(worst, (got - want).abs().max().item())
        rel = max(rel, ((got - want).norm() / want.norm()).item())
    ttnn.synchronize_device(mesh_device)
    recs = {}
    rt = ttnn.device.IsProgramRealtimeProfilerActive()

    def on_batch(batch):
        for r_ in batch.records:  # [runtime id] -> {chip: us}
            recs.setdefault(int(r_.runtime_id), {})[int(r_.chip_id)] = (
                (r_.end_timestamp - r_.start_timestamp) / r_.frequency / 1e3
            )

    hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch) if rt else None
    ids = []
    t0 = time.perf_counter()
    for _ in range(CALLS):
        i0 = ttnn._ttnn.get_device_operation_id()
        ttnn.deallocate(call())
        ids.append((i0, ttnn._ttnn.get_device_operation_id()))
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / CALLS * 1e6
    dev = ""
    if hdl is not None:
        time.sleep(0.5)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
        per = [max((recs.get(i, {}) or {0: 0}).values()) for a_, b_ in ids for i in range(a_, b_) if i in recs]
        if per:
            dev_us = sorted(per)[len(per) // 2]
            busiest = (G - 1 if topology == ttnn.Topology.Linear or G == 2 else G // 2) * Sb * H * 2
            dev = f", device {dev_us:7.1f} us (busiest chip, median), busiest link dir {busiest / dev_us / 1e3 / LINKS:5.1f} GB/s per link"
    print(
        f"FABRIC_RS {rows}x{cols} {FABRIC} axis {cluster_axis} {topology} {scheme} G {G} T {T} links {LINKS}: "
        f"{us:8.1f} us/call wall{dev}, max abs err {worst:.3g}, rel {rel:.2e}",
        flush=True,
    )
    # bf16 partials summed in bf16 along the chain (G - 1 roundings), vs the fp32 sum
    assert rel < 1e-2 and worst <= 0.0625 * G, (worst, rel)
