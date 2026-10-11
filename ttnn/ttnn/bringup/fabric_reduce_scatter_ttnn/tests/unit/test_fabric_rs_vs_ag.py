# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter vs fabric_all_gather at matched traffic: the reduce-scatter takes a [1, 1, G Sb, H] bf16
partial per chip to its [1, 1, Sb, H] block of the sum; the all-gather takes a [1, 1, Sb, H] shard per chip to the
[1, 1, G Sb, H] concatenation (dim -2). Over a line or a ring both move the same bytes per link direction (G - 1
blocks on a line's middle link, about G / 2 on a ring), so the gap is the reduce-scatter's extra work: reading the
local partial and adding in every hop.

Two all-gathers: fabric_all_gather_py (the Python op on the same planner as the reduce-scatter, explicit topology)
and ttnn.bringup.fabric_all_gather (the C++ op the model runs; no topology argument: a ring wherever the fabric
closes one, so it is timed only where that is the case's topology - on a FABRIC_2D 2 x 4 the axes are lines, the
whole mesh a snake ring).

Groups: each mesh axis (line; ring where the fabric closes it) and the whole mesh (snake line, snake ring). A group
the fabric cannot route is skipped. Both ops write into preallocated outputs. Timing as fabric_all_gather's device
time tests: CALLS back-to-back calls captured in one trace, replayed REPLAYS times, median replay / CALLS (host
dispatch excluded, the inter-call fence included). Each line reports us/call, the bytes one chip receives per second
((G - 1) Sb H 2 / t) and the busiest link direction's GB/s per link.

  scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_vs_ag.py -s -rs

RSAG_ROWS (Sb G per chip, comma list, default 5120,20480), RSAG_PAYLOAD (router payload bytes, default 8192, the GLM
spec), RSAG_LINKS (default 2), RSAG_CALLS (default 10), RSAG_REPLAYS (default 5)."""

import os
import statistics
import time

import pytest
import torch

import ttnn

ROWS = [int(v) for v in os.environ.get("RSAG_ROWS", "5120,20480").split(",")]
LINKS = int(os.environ.get("RSAG_LINKS", "2"))
CALLS = int(os.environ.get("RSAG_CALLS", "10"))
REPLAYS = int(os.environ.get("RSAG_REPLAYS", "5"))
H = 4096

CASES = [
    pytest.param(0, ttnn.Topology.Linear, id="axis0_line"),
    pytest.param(1, ttnn.Topology.Linear, id="axis1_line"),
    pytest.param(1, ttnn.Topology.Ring, id="axis1_ring"),
    pytest.param(None, ttnn.Topology.Linear, id="mesh_line"),
    pytest.param(None, ttnn.Topology.Ring, id="mesh_ring"),
]


def _device_params():
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = int(os.environ.get("RSAG_PAYLOAD", "8192"))
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": rc, "trace_region_size": 64 << 20}


def _time_per_call(mesh_device, issue):
    issue()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(CALLS):
        issue()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    samples = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) / CALLS)
    ttnn.release_trace(mesh_device, tid)
    return statistics.median(samples)


def _put(mesh_device, host):
    rows, cols = tuple(mesh_device.shape)
    t = ttnn.from_torch(
        host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    return ttnn.reshape(t, (1, 1, host.shape[-2], H))


def _group(r, c, rows, cols, cluster_axis):
    """(r, c)'s group as coords in row-major rank order, and (r, c)'s rank."""
    if cluster_axis == 0:
        return [(i, c) for i in range(rows)], r
    if cluster_axis == 1:
        return [(r, i) for i in range(cols)], c
    return [(i, j) for i in range(rows) for j in range(cols)], r * cols + c


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["2x4"])
@pytest.mark.parametrize("cluster_axis, topology", CASES)
def test_fabric_rs_vs_ag(mesh_device, cluster_axis, topology):
    from ttnn.bringup.fabric_all_gather_ttnn.fabric_all_gather_py import fabric_all_gather as ag_py
    from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

    rows, cols = tuple(mesh_device.shape)
    G = rows * cols if cluster_axis is None else (rows, cols)[cluster_axis]
    ring = topology == ttnn.Topology.Ring and G > 2
    for T in ROWS:
        T -= T % (32 * G)
        Sb = T // G
        torch.manual_seed(T)
        parts = torch.randn(rows, cols, T, H).bfloat16()
        shards = torch.randn(rows, cols, Sb, H).bfloat16()
        x_rs, x_ag = _put(mesh_device, parts), _put(mesh_device, shards)
        if cluster_axis is None:  # the C++ op's full mesh wants the input marked as sharded over dim 2
            ttnn.deallocate(x_ag)
            x_ag = ttnn.from_torch(
                shards.reshape(1, 1, rows * cols * Sb, H),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
            )
        kw = dict(cluster_axis=cluster_axis, topology=topology, num_links=LINKS)
        try:
            y_rs = fabric_reduce_scatter(x_rs, **kw)
            y_ag = ag_py(x_ag, dim=2, **kw)
        except ValueError as e:
            pytest.skip(f"unsupported on this fabric: {e}")

        rs_err = ag_err = 0.0
        for d, (a, b) in enumerate(zip(ttnn.get_device_tensors(y_rs), ttnn.get_device_tensors(y_ag))):
            grp, rank = _group(*divmod(d, cols), rows, cols, cluster_axis)
            want = sum(parts[i, j, rank * Sb : (rank + 1) * Sb].float() for i, j in grp)
            got = ttnn.to_torch(a).float().reshape(Sb, H)
            rs_err = max(rs_err, ((got - want).norm() / want.norm()).item())
            want = torch.cat([shards[i, j] for i, j in grp]).float()
            ag_err = max(ag_err, (ttnn.to_torch(b).float().reshape(G * Sb, H) - want).abs().max().item())

        t_rs = _time_per_call(mesh_device, lambda: fabric_reduce_scatter(x_rs, output=y_rs, **kw))
        t_ag = _time_per_call(mesh_device, lambda: ag_py(x_ag, dim=2, output=y_ag, **kw))
        timed = [("reduce_scatter", t_rs), ("all_gather py", t_ag)]
        if ring == (cluster_axis is None):  # the C++ op's own topology on this fabric
            y_c = ttnn.allocate_tensor_on_device(y_ag.spec, mesh_device)
            ttnn.bringup.fabric_all_gather(x_ag, 2, y_c, cluster_axis=cluster_axis, num_links=LINKS)
            c_err = max(
                (ttnn.to_torch(a).float() - ttnn.to_torch(b).float()).abs().max().item()
                for a, b in zip(ttnn.get_device_tensors(y_c), ttnn.get_device_tensors(y_ag))
            )
            assert c_err == 0.0, c_err
            t_c = _time_per_call(
                mesh_device,
                lambda: ttnn.bringup.fabric_all_gather(x_ag, 2, y_c, cluster_axis=cluster_axis, num_links=LINKS),
            )
            timed.append(("all_gather C++", t_c))
            ttnn.deallocate(y_c)
        block = Sb * H * 2
        busiest = (G / 2 if ring else G - 1) * block / LINKS
        for name, t in timed:
            print(
                f"RS_VS_AG {rows}x{cols} axis {cluster_axis} {'ring' if ring else 'line'} G {G} block {Sb}x{H}: "
                f"{name:14s} {t * 1e6:8.1f} us/call, in {(G - 1) * block / t / 1e9:6.1f} GB/s/chip, "
                f"busiest link {busiest / t / 1e9:5.1f} GB/s/link",
                flush=True,
            )
        print(
            f"RS_VS_AG {rows}x{cols} axis {cluster_axis} G {G} block {Sb}: RS / AG "
            + ", ".join(f"{n.split()[-1] if ' ' in n else n} {t_rs / t:.2f}" for n, t in timed[1:]),
            flush=True,
        )
        for t_ in (x_rs, x_ag, y_rs, y_ag):
            ttnn.deallocate(t_)
        assert rs_err < 1e-2, rs_err
        assert ag_err == 0.0, ag_err
