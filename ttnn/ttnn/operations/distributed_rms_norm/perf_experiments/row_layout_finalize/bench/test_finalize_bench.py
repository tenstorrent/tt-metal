# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated micro-bench: distributed_rms_norm root finalize, column layout (the op's current chain) vs compact
row / packed layouts with one fused sfpi pass. Single core, sharded L1 in/out, serialized latency loop (see the kernel
head). Correctness: every variant's inv_rms is compared BITWISE with the baseline's (same fp32 ops, same order) and
approximately with torch. Perf: zone FIN per rep (ns), measured, never asserted.

    FB_REPS=400 scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/row_layout_finalize/bench/test_finalize_bench.py -s
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # test-only; no global torch import under ttnn/
import ttnn

KERNEL = Path(__file__).parent / "finalize_bench_compute.cpp"
NAMES = {0: "base", 1: "col_fused", 2: "row_fused", 3: "packed", 4: "dense", 5: "floor_G", 6: "floor_1"}
ONE_TILE = {3, 4, 6}
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
RESULTS = Path(__file__).parent / "results.txt"


def slot_pos(V, G, q, e):
    """(row, col) inside the slot's 32x32 tile of record element e of slot q (logical coords)."""
    if V in (0, 1, 5):
        return e, 0
    if V == 2:
        return e // 16, e % 16
    if V in (3, 6):
        f = q // 4
        return (f // 2) * 16 + 4 * (q % 4) + e // 16, (f % 2) * 16 + e % 16
    # dense: slot q = sfpi unit q: face q // 8, rows 4 * ((q % 8) // 2) + e // 8, columns 2 * (e % 8) + parity
    f, u = q // 8, q % 8
    return (f // 2) * 16 + 4 * (u // 2) + e // 8, (f % 2) * 16 + 2 * (e % 8) + (u % 2)


def build_input(V, G, n, recs):
    """recs [n, G, 32] -> stacked tiles [(n * tiles_per_row) * 32, 32]."""
    tpr = 1 if V in ONE_TILE else G
    x = torch.ones(n * tpr * 32, 32, dtype=torch.float32)
    for t in range(n):
        for q in range(G):
            tile = t * tpr + (0 if tpr == 1 else q)
            for e in range(32):
                r, c = slot_pos(V, G, q, e)
                x[tile * 32 + r, c] = recs[t, q, e]
    return x, n * tpr


def read_output(V, G, n, out):
    res = torch.empty(n, 32, dtype=torch.float32)
    for t in range(n):
        for e in range(32):
            r, c = slot_pos(V, G, 0, e)
            res[t, e] = out[t * 32 + r, c]
    return res


def _zone_ns(seen):
    with open(_DEVICE_CSV) as f:
        lines = f.read().splitlines()
    freq = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    starts, ends = {}, {}
    for r in rows:
        if len(r) < 12 or r[7] in seen or r[10] != "FIN":
            continue
        (starts if r[11] == "ZONE_START" else ends).setdefault(r[3], []).append(int(r[5]))
    return {k: (max(ends[k]) - min(v)) * 1000.0 / freq for k, v in starts.items() if k in ends}


def _run_ids():
    if not os.path.exists(_DEVICE_CSV):
        return set()
    with open(_DEVICE_CSV) as f:
        lines = f.read().splitlines()
    return {ln.split(",")[7].strip() for ln in lines[2:] if ln.count(",") >= 11}


def run_variant(device, V, G, n, recs, fp32, reps, inv_w, eps):
    dt = ttnn.float32 if fp32 else ttnn.bfloat16
    x, tiles_in = build_input(V, G, n, recs)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def mc(rows):
        return ttnn.create_sharded_memory_config(
            shape=(rows, 32),
            core_grid=core,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    xt = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc(tiles_in * 32))
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([n * 32, 32]), dt, ttnn.TILE_LAYOUT, device, mc(n * 32))
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, math_approx_mode=False
    )
    if fp32:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=core,
        compile_time_args=[V, G, n, tiles_in],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps, bits(inv_w), bits(eps)])],
        config=cfg,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xt), ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    seen = _run_ids()
    ttnn.generic_op([xt, out], desc)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    z = _zone_ns(seen)
    res = read_output(V, G, n, ttnn.to_torch(out).to(torch.float32))
    return res, {k2: v / reps for k2, v in z.items()}


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
@pytest.mark.parametrize("G,n", [(4, 1), (4, 2), (2, 1)])
def test_finalize_bench(device, fp32, G, n):
    reps = int(os.environ.get("FB_REPS", "400"))
    torch.manual_seed(7 + G + n)
    W = 1792 * G
    inv_w, eps = 1.0 / W, 1e-6
    recs = torch.rand(n, G, 32) * 1000.0 + 1e-3
    recs[0, :, 0] = 0.0  # a zero-sum row (rsqrt(eps)) edge
    if not fp32:
        recs = recs.to(torch.bfloat16).to(torch.float32)
    variants = [int(v) for v in os.environ.get("FB_VARIANTS", "0,1,2,3,4,5,6").split(",")]
    ref = None
    lines = []
    for V in variants:
        res, z = run_variant(device, V, G, n, recs, fp32, reps, inv_w, eps)
        ns = statistics.median(list(z.values())) if z else float("nan")
        tag = ""
        if V in (0, 1, 2, 3, 4):
            golden = torch.rsqrt(recs.sum(1) * inv_w + eps)
            rel = ((res - golden).abs() / golden.abs()).max().item()
            assert rel < (1e-5 if fp32 else 2e-2), f"{NAMES[V]} rel {rel}"
            if V == 0:
                ref = res
            elif ref is not None:
                same = torch.equal(res.view(torch.int32), ref.view(torch.int32))
                ndiff = int((res.view(torch.int32) != ref.view(torch.int32)).sum())
                tag = "BITWISE==base" if same else f"DIFF vs base ({ndiff}/{res.numel()})"
                assert same, f"{NAMES[V]} not bit-identical to base"
            tag = f"rel={rel:.2e} {tag}"
        per = " ".join(f"{k2}={v:.1f}" for k2, v in sorted(z.items()))
        line = f"G={G} n={n} {'fp32' if fp32 else 'bf16'}dest {NAMES[V]:10s} {ns:8.1f} ns/rep  [{per}]  {tag}"
        print(line)
        lines.append(line)
    with open(RESULTS, "a") as f:
        f.write("\n".join(lines) + "\n")
