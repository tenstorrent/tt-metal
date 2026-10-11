# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's full-mesh reduce-scatters at the model's shape: what the model runs (common.scatter_rows: the
shared expert's and the dense MLP's output, fabric_reduce_scatter on axis 0 then axis 1) vs one call over the whole
mesh (a snake line, a snake ring). Per chip [1, 1, 5120, 4096] bf16 partials (a 5120-token chunk) -> the chip's
[640, 4096] block of the 8-chip sum, row-major; all variants checked against the torch sum and each other.

Device time per variant from the program real-time profiler: per call, each chip's summed program durations, the
busiest chip, median of RS_CALLS calls (the two-stage variant is one "call" of both programs).

  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_model_shapes.py -s

RS_TRACE=1 also times a traced replay of each variant (no host dispatch). RS_PAYLOAD (router payload bytes, default 8192 as the GLM spec), RS_ROWS (default 5120), RS_LINKS (default 2, the model's GLM_MOE_LINKS), RS_CALLS (default 20)."""

import os
import statistics
import time

import pytest
import torch

import ttnn

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
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["2x4"])
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
    ref = parts.float().sum((0, 1))  # [ROWS, H]; chip d (row-major) owns rows d ROWS/n ..
    Sb = ROWS // n

    def two_stage():  # common.scatter_rows (GLM_SCATTER_OP=fabric_bf16, the default)
        a = fabric_reduce_scatter(x, cluster_axis=0, num_links=LINKS)
        b = fabric_reduce_scatter(a, cluster_axis=1, num_links=LINKS)
        ttnn.deallocate(a)
        return b

    variants = {
        "current: axis 0 + axis 1 (2 calls)": two_stage,
        "mesh line (1 call)": lambda: fabric_reduce_scatter(
            x, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=LINKS
        ),
        "mesh ring (1 call)": lambda: fabric_reduce_scatter(
            x, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=LINKS
        ),
    }
    if os.environ.get("RS_VARIANTS"):  # comma list of name prefixes: current, mesh line, mesh ring
        keep = [v.strip() for v in os.environ["RS_VARIANTS"].split(",")]
        variants = {k: v for k, v in variants.items() if any(k.startswith(p_) for p_ in keep)}
    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    outs, rows_out = {}, []
    for name, f in variants.items():
        y = f()  # warm-up / compile, and the result for the check
        host = torch.cat([ttnn.to_torch(t).float().reshape(Sb, H) for t in ttnn.get_device_tensors(y)])
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
            dev = statistics.median(per_call)
            per_prog = [statistics.median(p[k] for p in progs if len(p) > k) for k in range(max(map(len, progs)))]
        traced = None
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
        rows_out.append((name, dev, per_prog, wall, err, rel))
        print(
            f"RS_MODEL {name:36s} device {dev:7.1f} us"
            + (f" (per program {', '.join(f'{v:.1f}' for v in per_prog)})" if per_prog and len(per_prog) > 1 else "")
            + f", wall {wall:7.1f} us/call, max abs err {err:.3g}, rel {rel:.2e}",
            flush=True,
        )
    base = outs.get("current: axis 0 + axis 1 (2 calls)")
    for name, host in outs.items() if base is not None else []:
        d = (host - base).abs().max().item()
        print(f"RS_MODEL {name:36s} vs current: max abs diff {d:.3g}", flush=True)
    for name, dev, _, _, err, rel in rows_out:
        assert rel < 1e-2 and err <= 0.5, (name, err, rel)
