# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the duplex roofline probe: read-only / write-only / both, same bytes.

    RW_SHAPES=8192x2304,8192x1024 RW_BLOCK=72 scripts/run_safe_pytest.sh <this file>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics

import ttnn

from ttnn.operations.rms_norm_ttnn.perf_experiments.rw_overlap.duplex import duplex

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_TRIALS = int(os.environ.get("RW_TRIALS", "3"))


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def test_duplex():
    import torch

    shapes = os.environ.get("RW_SHAPES", "8192x2304").split(",")
    blocks = [int(v) for v in os.environ.get("RW_BLOCK", "72").split(",")]
    ring = int(os.environ.get("RW_RING", "4"))
    core_caps = [int(v) for v in os.environ.get("RW_CORES", "0").split(",")]
    device = ttnn.open_device(device_id=0)
    try:
        for s in shapes:
            H, W = (int(v) for v in s.split("x"))
            torch.manual_seed(0)
            t = torch.randn(1, 1, H, W, dtype=torch.float32).to(torch.bfloat16)
            x = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            nbytes = H * W * 2
            for cap in core_caps:
                for block in blocks:
                    res = {}
                    for mode in ("read", "write", "both"):
                        duplex(x, out, mode=mode, block=block, ring_pages=ring, num_cores_cap=cap)
                        ttnn.synchronize_device(device)
                        _read_kernel_ns(device)
                        samples = []
                        for _ in range(N_TRIALS):
                            duplex(x, out, mode=mode, block=block, ring_pages=ring, num_cores_cap=cap)
                            ttnn.synchronize_device(device)
                            v = _read_kernel_ns(device)
                            if v is not None:
                                samples.append(v)
                        med = statistics.median(samples)
                        moved = nbytes * (2 if mode == "both" else 1)
                        res[mode] = med
                        print(
                            f"RESULT duplex {s:10s} cores={cap:3d} blk={block:3d} mode={mode:5s} "
                            f"median={med:9.0f} ns min={min(samples):9.0f} GB/s={moved/med:7.1f}"
                        )
                    print(
                        f"RESULT duplex {s:10s} cores={cap:3d} blk={block:3d} SUMMARY serial={res['read']+res['write']:9.0f} "
                        f"max(r,w)={max(res['read'],res['write']):9.0f} both={res['both']:9.0f} "
                        f"eff={(res['read']+res['write'])/res['both']:.3f}"
                    )
            ttnn.deallocate(x)
            ttnn.deallocate(out)
    finally:
        ttnn.close_device(device)
