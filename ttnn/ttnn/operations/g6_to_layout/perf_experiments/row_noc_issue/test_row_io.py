# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row I/O bench: correctness of every issue variant (check mode) and its cost (time mode).

    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    D=ttnn/ttnn/operations/g6_to_layout/perf_experiments/row_noc_issue
    scripts/run_safe_pytest.sh $D/test_row_io.py::test_check
    scripts/run_safe_pytest.sh $D/test_row_io.py::test_perf     # RIO_SET=core|grid|rot|all

Time mode reports, per variant: DEVICE KERNEL DURATION (median of RIO_LAUNCHES launches after one
warm-up) and, from the kernel's own wall-clock counters, issue cycles per transaction (median and
max over cores) and the core's whole loop in cycles (max over cores).
"""

import os
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import row_io_bench as rb  # noqa: E402

_KEY = "DEVICE KERNEL DURATION [ns]"
LAUNCHES = int(os.environ.get("RIO_LAUNCHES", "3"))
SET = os.environ.get("RIO_SET", "all")
VARIANT_SEL = os.environ.get("RIO_VARIANTS", "")


def _mem(buf):
    return ttnn.L1_MEMORY_CONFIG if buf == "L1" else ttnn.DRAM_MEMORY_CONFIG


def _tensors(device, h, w, buf_in, buf_out, seed=0):
    torch.manual_seed(seed)
    x = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    inp = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_mem(buf_in)
    )
    out = ttnn.from_torch(
        torch.zeros((h, w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_mem(buf_out),
    )
    grid = device.compute_with_storage_grid_size()
    stats = ttnn.from_torch(
        torch.zeros((grid.x * grid.y, rb.STATS_WORDS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x, inp, out, stats


def _durations(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_KEY)
            if entry is not None:
                vals.append(float(entry.duration))
    return vals


def _stats(stats, cores):
    s = ttnn.to_torch(stats).to(torch.int64)[:cores]
    assert bool((s[:, 7] == 0x600DF00D).all()), "stats page missing"
    return s


# ------------------------------------------------------------------ correctness
# (H, W, block_bytes): aligned, W % 32 != 0, H % 32 != 0, H < 32, several column blocks, full rows.
_CHECK_SHAPES = [
    (32, 512, 64),
    (64, 2560, 1024),
    (50, 100, 64),
    (7, 48, 64),
    (96, 256, 512),
    (40, 100, 256),
    (130, 32, 64),
]
_FULL_ROW_DRAM = [(96, 256, 512), (40, 96, 256), (200, 1024, 2048)]


def test_check(device):
    failures, runs = [], 0
    for buf in ("DRAM", "L1"):
        for h, w, bb in _CHECK_SHAPES:
            for variant in ("accessor", "cursor", "state", "bank_major", "state_bulk"):
                for rotate, cores in ((0, 1), (0, 110), (1, 110), ("spread", 110)):
                    if True:
                        x, inp, out, stats = _tensors(device, h, w, buf, buf)
                        desc, n = rb.build(
                            device,
                            inp,
                            out,
                            stats,
                            mode="check",
                            variant=variant,
                            block_bytes=bb,
                            num_cores=cores,
                            rotate=rotate,
                        )
                        ttnn.generic_op([inp, out, stats], desc)
                        got = ttnn.to_torch(out)
                        runs += 1
                        if not torch.equal(got.view(torch.int16), x.view(torch.int16)):
                            bad = (got.view(torch.int16) != x.view(torch.int16)).nonzero()
                            failures.append((buf, h, w, bb, variant, rotate, cores, bad[:4].tolist()))
                        for t in (inp, out, stats):
                            ttnn.deallocate(t)
    for h, w, bb in _FULL_ROW_DRAM:
        for rotate in (0, 1):
            x, inp, out, stats = _tensors(device, h, w, "DRAM", "DRAM")
            desc, n = rb.build(
                device,
                inp,
                out,
                stats,
                mode="check",
                variant="run_coalesce",
                block_bytes=bb,
                num_cores=110,
                rotate=rotate,
            )
            ttnn.generic_op([inp, out, stats], desc)
            got = ttnn.to_torch(out)
            runs += 1
            if not torch.equal(got.view(torch.int16), x.view(torch.int16)):
                failures.append(("DRAM", h, w, bb, "run_coalesce", rotate, 110))
            for t in (inp, out, stats):
                ttnn.deallocate(t)
    print(f"\nRIO_CHECK runs={runs} failures={len(failures)}")
    for f in failures:
        print("RIO_CHECK_FAIL", f)
    assert not failures


# ------------------------------------------------------------------ timing
def _scenarios(grid_cores):
    out = []
    # (name, mode, buf, H, W, block_bytes, cores, rotate, variants)
    base = ["accessor", "cursor", "state", "bank_major", "state_bulk"]
    if SET in ("core", "all"):
        for mode in ("write", "read"):
            for buf in ("DRAM", "L1"):
                for w in (32, 256, 1024):
                    out.append((f"1core_{mode}_{buf}_{2*w}B", mode, buf, 32 * 16, w, 2 * w, 1, 0, base))
    if SET in ("grid", "all"):
        for mode in ("write", "read"):
            for buf in ("DRAM", "L1"):
                for w in (32, 256):
                    vs = base + (["run_coalesce"] if mode == "write" and buf == "DRAM" else [])
                    out.append(
                        (f"grid_{mode}_{buf}_{2*w}B", mode, buf, 32 * grid_cores * 3, w, 2 * w, grid_cores, 0, vs)
                    )
    if SET in ("rot", "all"):
        for mode in ("write", "read"):
            for w in (32, 256):
                out.append(
                    (
                        f"grid_{mode}_DRAM_{2*w}B_rot",
                        mode,
                        "DRAM",
                        32 * grid_cores * 3,
                        w,
                        2 * w,
                        grid_cores,
                        1,
                        ["accessor", "state"],
                    )
                )
    if SET == "floor":
        out += [
            (
                "grid_read_L1_512B",
                "read",
                "L1",
                32 * grid_cores * 3,
                256,
                512,
                grid_cores,
                0,
                ["accessor", "floor48", "floor72", "floor96"],
            ),
            ("c49r", "read", "L1", 32768, 128, 256, grid_cores, 0, ["accessor", "floor48"]),
            (
                "grid_read_DRAM_512B",
                "read",
                "DRAM",
                32 * grid_cores * 3,
                256,
                512,
                grid_cores,
                0,
                ["accessor", "floor72", "floor96"],
            ),
            (
                "grid_write_L1_512B",
                "write",
                "L1",
                32 * grid_cores * 3,
                256,
                512,
                grid_cores,
                0,
                ["accessor", "floor72"],
            ),
        ]
    if SET == "pace":
        out += [
            (
                "c49r",
                "read",
                "L1",
                32768,
                128,
                256,
                grid_cores,
                0,
                ["state_bulk_pace8", "accessor", "state_bulk_pace4", "state_bulk"],
            ),
            ("c49r_1core", "read", "L1", 32 * 16, 128, 256, 1, 0, ["state_bulk_pace8", "state_bulk_pace4"]),
            (
                "grid_read_L1_64B",
                "read",
                "L1",
                32 * grid_cores * 3,
                32,
                64,
                grid_cores,
                0,
                ["state_bulk_pace8", "accessor", "state_bulk"],
            ),
            (
                "grid_read_L1_128B",
                "read",
                "L1",
                32 * grid_cores * 3,
                64,
                128,
                grid_cores,
                0,
                ["accessor", "state_bulk"],
            ),
            ("grid_read_L1_256B_55c", "read", "L1", 32 * 55 * 6, 128, 256, 55, 0, ["accessor", "state_bulk"]),
        ]
    if SET == "c49":
        out += [
            (
                "c49r",
                "read",
                "L1",
                32768,
                128,
                256,
                grid_cores,
                0,
                ["state", "accessor", "state_bulk", "cursor", "accessor"],
            ),
            ("c49r_rot", "read", "L1", 32768, 128, 256, grid_cores, 1, ["state", "accessor", "state_bulk"]),
            (
                "c49r_spread",
                "read",
                "L1",
                32768,
                128,
                256,
                grid_cores,
                "spread",
                ["state_bulk", "accessor", "state", "cursor"],
            ),
            ("c49w_spread", "write", "L1", 32768, 128, 256, grid_cores, "spread", ["state_bulk", "accessor"]),
            ("c49w", "write", "L1", 32768, 128, 256, grid_cores, 0, ["state_bulk", "accessor"]),
            ("c277r_spread", "read", "DRAM", 8192, 256, 512, grid_cores, "spread", ["accessor", "state_bulk"]),
            ("c296r_spread", "read", "DRAM", 32768, 32, 64, grid_cores, "spread", ["accessor", "state_bulk"]),
            ("c297w_spread", "write", "DRAM", 32768, 32, 64, grid_cores, "spread", ["accessor", "state_bulk"]),
            ("c49r_1core", "read", "L1", 32 * 16, 128, 256, 1, 0, ["accessor", "state_bulk"]),
            ("c49r_16core", "read", "L1", 32 * 16 * 10, 128, 256, 16, 0, ["accessor", "state_bulk"]),
        ]
    if SET == "cells":
        # Shapes of the measured cells, the op's own work splits approximated.
        out += [
            ("c2025w", "write", "DRAM", 32, 512, 64, 16, 0, base),
            ("c49r", "read", "L1", 32768, 128, 256, grid_cores, 0, base),
            ("c277r", "read", "DRAM", 8192, 256, 512, grid_cores, 0, base),
            ("c277r_rot", "read", "DRAM", 8192, 256, 512, grid_cores, 1, ["accessor", "state"]),
            ("c296r", "read", "DRAM", 32768, 32, 64, grid_cores, 0, ["accessor", "state"]),
            ("c297w", "write", "DRAM", 32768, 32, 64, grid_cores, 0, ["accessor", "state", "run_coalesce"]),
        ]
    return out


def test_perf(device):
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    rows = []
    for name, mode, buf, h, w, bb, cores, rotate, variants in _scenarios(grid_cores):
        if VARIANT_SEL:
            variants = [v for v in variants if v in VARIANT_SEL.split(",")]
        x, inp, out, stats = _tensors(device, h, w, buf, buf)
        for variant in variants:
            desc, n = rb.build(
                device, inp, out, stats, mode=mode, variant=variant, block_bytes=bb, num_cores=cores, rotate=rotate
            )
            ttnn.generic_op([inp, out, stats], desc)  # warm-up / JIT
            ttnn.synchronize_device(device)
            _durations(device)
            for _ in range(LAUNCHES):
                ttnn.generic_op([inp, out, stats], desc)
                ttnn.synchronize_device(device)
            ns = _durations(device)
            s = _stats(stats, n)
            txn = s[:, 1] if mode == "read" else s[:, 2]
            cyc = s[:, 3] if mode == "read" else s[:, 4]
            per = (cyc.double() / txn.clamp(min=1).double()).tolist()
            rows.append(
                (
                    name,
                    variant,
                    statistics.median(ns) if ns else float("nan"),
                    statistics.median(per),
                    max(per),
                    int(txn.max()),
                    int(s[:, 5].max()),
                    n,
                )
            )
            r = rows[-1]
            print(
                f"RIO {r[0]:<26} {r[1]:<13} ns={r[2]:>9.0f}  issue_cyc/txn med={r[3]:6.1f} max={r[4]:6.1f}  "
                f"txn/core={r[5]:>4}  loop_cyc_max={r[6]:>7}  cores={r[7]}  all_ns={[round(v) for v in ns]}",
                flush=True,
            )
        for t in (inp, out, stats):
            ttnn.deallocate(t)
