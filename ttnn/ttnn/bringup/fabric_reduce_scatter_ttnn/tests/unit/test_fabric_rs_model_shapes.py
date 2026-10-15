# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's reduce-scatters at the model's shape, per chip [1, 1, 5120, 4096] bf16 partials (a 5120-token chunk):

- common.scatter_rows (the shared expert's and the dense MLP's output): one call over the whole mesh as a snake ring
  (the default, GLM_SCATTER_OP=fabric_ring), else axis 0 then axis 1; also a snake line. Each chip gets its row-major
  [5120 / n, 4096] block of the n-chip sum; all checked against the torch sum and each other. The ring also with
  final="bwd_port" (each chip's own block summed by its backward port, no final cores).
- mla_attention o_proj (mla_attention.py:470): axis 1, chip (r, c) gets block c of the sum over its mesh row.

Device time per variant from the program real-time profiler (when active): per call, each chip's summed program
durations, the busiest chip, median of RS_CALLS calls (the two-stage variant is one "call" of both programs).

  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_model_shapes.py -s

RS_MESH (default 2x4; 2x2 on a QuietBox), RS_TRACE=1 also times a traced replay of each variant (no host dispatch),
RS_PAYLOAD (router payload bytes, default 8192 as the GLM spec), RS_ROWS (default 5120), RS_LINKS (default 2, the
model's GLM_MOE_LINKS), RS_CALLS (default 20), RS_LAND_KIB (landing buffer per port, default 256), RS_VARIANTS (comma list of name prefixes)."""

import os
import statistics
import time

import pytest
import torch

import ttnn

MESH = tuple(int(v) for v in os.environ.get("RS_MESH", "2x4").split("x"))
ROWS = int(os.environ.get("RS_ROWS", "5120"))
LINKS = int(os.environ.get("RS_LINKS", "2"))
CALLS = int(os.environ.get("RS_CALLS", "20"))
H = 4096


def _device_params():
    """FABRIC_2D with the model's router payload (GLM spec fabric_payload_bytes 8192: the chunk size)."""
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = int(os.environ.get("RS_PAYLOAD", "8192"))
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": rc, "trace_region_size": 64 << 20}


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True, ids=[f"{MESH[0]}x{MESH[1]}"])
def test_fabric_rs_model_shapes(mesh_device):
    from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

    rows, cols = tuple(mesh_device.shape)
    n = rows * cols
    torch.manual_seed(3)
    parts = torch.randn(rows, cols, ROWS, H).bfloat16()
    x = ttnn.from_torch(
        parts,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x = ttnn.reshape(x, (1, 1, ROWS, H))
    ref_mesh = parts.float().sum((0, 1))  # [ROWS, H]; chip d (row-major) owns rows d ROWS/n ..
    row_sum = parts.float().sum(1)  # [rows, ROWS, H]; axis 1: chip (r, c) owns rows c ROWS/cols .. of row r's sum
    ref_axis1 = torch.cat(
        [row_sum[r, c * (ROWS // cols) : (c + 1) * (ROWS // cols)] for r in range(rows) for c in range(cols)]
    )

    def two_stage():  # common.scatter_rows, GLM_SCATTER_OP=fabric_bf16 (or a mesh whose snake does not close)
        a = fabric_reduce_scatter(x, cluster_axis=0, num_links=LINKS)
        b = fabric_reduce_scatter(a, cluster_axis=1, num_links=LINKS)
        ttnn.deallocate(a)
        return b

    variants = {  # name: (call, reference of the row-major concatenation of every chip's output)
        "current: axis 0 + axis 1 (2 calls)": (two_stage, ref_mesh),
        "mesh line (1 call)": (
            lambda: fabric_reduce_scatter(x, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=LINKS),
            ref_mesh,
        ),
        "mesh ring (1 call)": (
            lambda: fabric_reduce_scatter(x, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=LINKS),
            ref_mesh,
        ),
        "mesh ring, final on bwd port": (
            lambda: fabric_reduce_scatter(
                x, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=LINKS, final="bwd_port"
            ),
            ref_mesh,
        ),
        "mesh ring, final on bwd port, L1 landing": (
            lambda: fabric_reduce_scatter(
                x,
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=LINKS,
                final="bwd_port_l1",
                land_kib=int(os.environ.get("RS_LAND_KIB", "256")),
            ),
            ref_mesh,
        ),
        "attention o_proj: axis 1": (lambda: fabric_reduce_scatter(x, cluster_axis=1, num_links=LINKS), ref_axis1),
    }
    if os.environ.get("RS_VARIANTS"):
        keep = [v.strip() for v in os.environ["RS_VARIANTS"].split(",")]
        variants = {k: v for k, v in variants.items() if any(k.startswith(p_) for p_ in keep)}
    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    if not rt:
        print("RS_MODEL the program real-time profiler is not active: wall time only", flush=True)
    outs, rows_out = {}, []
    for name, (f, ref) in variants.items():
        y = f()  # warm-up / compile, and the result for the check
        host = torch.cat([ttnn.to_torch(t).float().reshape(-1, H) for t in ttnn.get_device_tensors(y)])
        ttnn.deallocate(y)
        outs[name] = host
        err = (host - ref).abs().max().item()
        rel = ((host - ref).norm() / ref.norm()).item()
        ttnn.synchronize_device(mesh_device)
        recs = {}

        def on_batch(batch):
            for r_ in batch.records:
                recs.setdefault(int(r_.runtime_id), {})[int(r_.chip_id)] = (
                    (r_.end_timestamp - r_.start_timestamp) / r_.frequency / 1e3
                )

        hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch) if rt else None
        ids = []
        t0 = time.perf_counter()
        for _ in range(CALLS):
            i0 = ttnn._ttnn.get_device_operation_id()
            ttnn.deallocate(f())
            ids.append((i0, ttnn._ttnn.get_device_operation_id()))
        ttnn.synchronize_device(mesh_device)
        wall = (time.perf_counter() - t0) / CALLS * 1e6
        dev, per_prog = None, None
        if hdl is not None:
            time.sleep(0.5)
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
            per_call, progs = [], []
            for a_, b_ in ids:
                chips = {}
                prog_us = []
                for i in range(a_, b_):
                    if i in recs:
                        prog_us.append(max(recs[i].values()))
                        for c, us in recs[i].items():
                            chips[c] = chips.get(c, 0.0) + us
                if chips:
                    per_call.append(max(chips.values()))
                    progs.append(prog_us)
            if per_call:
                dev = statistics.median(per_call)
                per_prog = [statistics.median(p[k] for p in progs if len(p) > k) for k in range(max(map(len, progs)))]
        if os.environ.get("RS_TRACE") == "1":  # replay from a trace: no host dispatch, chips start in step
            tr = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            yt = f()
            ttnn.end_trace_capture(mesh_device, tr, cq_id=0)
            ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(CALLS):
                ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            traced = (time.perf_counter() - t0) / CALLS * 1e6
            ttnn.release_trace(mesh_device, tr)
            ttnn.deallocate(yt)
            print(f"RS_MODEL {name:36s} traced replay {traced:7.1f} us/call", flush=True)
        rows_out.append((name, err, rel))
        print(
            f"RS_MODEL {rows}x{cols} {name:36s} "
            + (f"device {dev:7.1f} us" if dev is not None else "device    n/a   ")
            + (f" (per program {', '.join(f'{v:.1f}' for v in per_prog)})" if per_prog and len(per_prog) > 1 else "")
            + f", wall {wall:7.1f} us/call, max abs err {err:.3g}, rel {rel:.2e}",
            flush=True,
        )
    base = outs.get("current: axis 0 + axis 1 (2 calls)")
    for name, host in outs.items() if base is not None else []:
        if variants[name][1] is ref_mesh:
            d = (host - base).abs().max().item()
            print(f"RS_MODEL {name:36s} vs current: max abs diff {d:.3g}", flush=True)
    for name, err, rel in rows_out:
        assert rel < 1e-2 and err <= 0.5, (name, err, rel)
