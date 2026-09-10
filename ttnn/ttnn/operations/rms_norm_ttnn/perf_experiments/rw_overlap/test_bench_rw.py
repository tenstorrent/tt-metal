# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rw_overlap bake-off: pipeline levers against the shipped read/write schedule.

Runs the op out of THIS DIRECTORY's private clone (descriptor + kernels); the
shipped op is never touched.  A variant is a dict of module-level knobs applied
to the cloned descriptor before each build, so every candidate is the SAME
kernels + the SAME user precision config (HiFi2 / fp32_dest_acc_en=False /
math_approx_mode=False, bf16 in and out) and only the read/write pipeline shape
changes.

    RW_VARIANTS=base,depth3,sub8 RW_CASES=FOCUS scripts/run_safe_pytest.sh <this file>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics

import ttnn

from ttnn.operations.rms_norm_ttnn.perf_experiments.rw_overlap.rms_norm_ttnn import (  # noqa: E402
    rms_norm_ttnn,
    torch_rms_norm_ttnn,
)
from ttnn.operations.rms_norm_ttnn.perf_experiments.rw_overlap import (  # noqa: E402
    rms_norm_ttnn_program_descriptor as PD,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_TRIALS = int(os.environ.get("RW_TRIALS", "3"))

# name -> (H, W)  -- all DRAM INTERLEAVED, bf16, TILE, gamma only.
CASES = {
    "FOCUS": (8192, 2304),  # the perf-flagged case: RESIDENT, WT_CHUNK=72, depth 2
    "W1024": (8192, 1024),
    "W5120": (8192, 5120),  # ROW_RESIDENT, WT_CHUNK=32, NUM_W_CHUNKS=5
    "W7168": (8192, 7168),  # ROW_RESIDENT, WT_CHUNK=56, NUM_W_CHUNKS=4
    "SMALL": (32, 1024),  # latency-bound: 32 tile-rows over the grid
}

VARIANTS = {
    "base": {},
    # ---- lever 1: deeper activation rings (reader runs further ahead) ----
    "depth3": {"CB_DEPTH_CANDIDATES": (3, 2)},
    "depth4": {"CB_DEPTH_CANDIDATES": (4, 3, 2)},
    "depth3l90": {"CB_DEPTH_CANDIDATES": (3, 2), "L1_SAFETY_FRACTION": 0.92},
    "depth4l90": {"CB_DEPTH_CANDIDATES": (4, 3, 2), "L1_SAFETY_FRACTION": 0.92},
    # ---- lever 2: finer WRITE drain granularity (start draining sooner) ----
    "sub8": {"RW_WR_SUBROW": 8},
    "sub16": {"RW_WR_SUBROW": 16},
    "sub24": {"RW_WR_SUBROW": 24},
    "sub36": {"RW_WR_SUBROW": 36},
    # ---- lever 2b: COARSER transaction group (the other direction) ----
    "txn0": {"DM_TXN_ROWS_MAX": 0},
    "txn2": {"DM_TXN_ROWS_MAX": 2},
    # ---- lever 3: force >= N row-blocks per core (restore the cross-block pipeline) ----
    "minblk2": {"RW_MIN_BLOCKS": 2},
    "minblk3": {"RW_MIN_BLOCKS": 3},
    "minblk2d3": {"RW_MIN_BLOCKS": 2, "CB_DEPTH_CANDIDATES": (3, 2)},
    # ---- combined best-of ----
    "depth3sub8": {"CB_DEPTH_CANDIDATES": (3, 2), "RW_WR_SUBROW": 8},
    # ---- CEILING PROBES (WRONG programs; pcc is garbage by construction) ----
    # `dmonly` is the op with the compute payload and the per-channel reads peeled
    # away: what is left is exactly the x-read + out-write pipeline plus every CB
    # handshake and barrier.  Compare it to duplex.py's `both` floor, which moves
    # the same bytes with NO synchronization at all.
    "abl_dmonly": {"RW_ABL_COMPUTE": 1, "RW_ABL_PER_CHANNEL": 1},
    "abl_dmonly_d3": {
        "RW_ABL_COMPUTE": 1,
        "RW_ABL_PER_CHANNEL": 1,
        "CB_DEPTH_CANDIDATES": (3, 2),
        "L1_SAFETY_FRACTION": 0.92,
    },
    "abl_dmonly_sub24": {"RW_ABL_COMPUTE": 1, "RW_ABL_PER_CHANNEL": 1, "RW_WR_SUBROW": 24},
    "abl_readonly": {"RW_ABL_COMPUTE": 1, "RW_ABL_PER_CHANNEL": 1, "RW_ABL_WRITE": 1},
    "abl_writeonly": {"RW_ABL_COMPUTE": 1, "RW_ABL_PER_CHANNEL": 1, "RW_ABL_READ_X": 1},
    "abl_none": {"RW_ABL_COMPUTE": 1, "RW_ABL_PER_CHANNEL": 1, "RW_ABL_READ_X": 1, "RW_ABL_WRITE": 1},
}


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


def _cfg():
    """THE USER'S PRECISION CONTRACT -- identical for every variant, never a lever."""
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def build(device, name):
    import torch

    H, W = CASES[name]
    torch.manual_seed(0)
    tx = torch.randn(1, 1, H, W, dtype=torch.float32).to(torch.bfloat16)
    tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {
        "epsilon": 1e-12,
        "weight": g,
        "compute_kernel_config": _cfg(),
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    expected = torch_rms_norm_ttnn(tx.float(), epsilon=1e-12, weight=tg.float())
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, [x, g]


def measure(device, name):
    import torch

    run, expected, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p = pcc(got, expected)
    del out, got
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_TRIALS):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return statistics.median(samples), min(samples), p


def test_bench():
    labels = os.environ.get("RW_VARIANTS", "base,depth3,sub8").split(",")
    names = os.environ.get("RW_CASES", "FOCUS").split(",")
    device = ttnn.open_device(device_id=0)
    RES = {}
    try:
        for label in labels:
            knobs = VARIANTS[label]
            saved = {k: getattr(PD, k) for k in knobs}
            for k, v in knobs.items():
                setattr(PD, k, v)
            try:
                for name in names:
                    try:
                        med, lo, p = measure(device, name)
                    except Exception as e:  # a knob may not fit L1 on some case
                        RES[(name, label)] = (float("nan"), float("nan"), float("nan"))
                        print(f"RESULT.err {name:6s} {label:11s} {type(e).__name__}: {str(e)[:140]}")
                        continue
                    RES[(name, label)] = (med, lo, p)
                    print(f"RESULT {name:6s} {label:11s} median={med:9.0f} min={lo:9.0f} pcc={p:.7f}")
            finally:
                for k, v in saved.items():
                    setattr(PD, k, v)
    finally:
        ttnn.close_device(device)

    base = labels[0]
    print(f"RESULT === speedup vs {base} (>1 = faster; median ns) ===")
    for name in names:
        b = RES[(name, base)][0]
        row = f"{name:6s}"
        for label in labels:
            m = RES[(name, label)][0]
            row += f" {label}={b / m:.3f}" if m == m else f" {label}=ERR"
        print("RESULT " + row)
