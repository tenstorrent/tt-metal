# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""STEP A of g6_clone perf1 stagger_placement: is the per-column kernel start offset a real late start
or a per-tile wall-clock offset? See kernels/clock_skew_probe.cpp for the protocol.

Run a copy from generated/ (a test file under ttnn/ imports ttnn twice), e.g. via run_clock.sh:
    SKEW_ROOT=<this dir> scripts/run_safe_pytest.sh --run-all <lane>/test_clock_skew.py
Lines starting SKEW_ carry the results; the full record goes to $SKEW_OUT (JSON) when set.

Physical (NOC0) column of a logical core on this p150b, as the profiler reports it: logical x 0..5 ->
x1..x6, 6..10 -> x11..x15; logical y 0..9 -> y2..y11.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import ttnn

ROOT = Path(os.environ.get("SKEW_ROOT", Path(__file__).resolve().parent))
KERNEL = ROOT / "kernels" / "clock_skew_probe.cpp"
CB_SCRATCH = 0
PAGE_WORDS = 16
LAUNCHES = 4
PHYS_X = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]
REF = (6, 1)  # physical x11 y3, the earliest-starting core of every capture


def _phys(core):
    return (PHYS_X[core[0]], core[1] + 2)


def _run(device, cores, both_ways, risc):
    """cores[0] is R. Returns per-launch lists of records."""
    import torch

    n = len(cores)
    virt = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y in cores]
    out_host = torch.zeros((2 * n, PAGE_WORDS), dtype=torch.int32)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=ranges, initial_value=0) for i in range(4)]
    config = ttnn.WriterConfigDescriptor() if risc == "brisc" else ttnn.ReaderConfigDescriptor()
    results = []
    for _ in range(LAUNCHES):
        out = ttnn.from_torch(
            out_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rt = ttnn.RuntimeArgs()
        others = [v for c in virt[1:] for v in (c.x, c.y)]
        for idx, (x, y) in enumerate(cores):
            common = [0 if idx == 0 else 1, idx, n, out.buffer_address(), virt[0].x, virt[0].y, both_ways]
            rt[x][y] = common + (others if idx == 0 else [])
        cb = ttnn.CBDescriptor(
            total_size=2 * n * PAGE_WORDS * 4,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCRATCH, data_format=ttnn.uint32, page_size=64)
            ],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=ranges,
            compile_time_args=[CB_SCRATCH] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            config=config,
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=sems, cbs=[cb])
        ttnn.generic_op([out, out], desc)
        ttnn.synchronize_device(device)
        words = ttnn.to_torch(out).to(torch.int64) & 0xFFFFFFFF
        results.append(words)
        ttnn.deallocate(out)
    return results


def _u64(row, slot):
    return int(row[2 * slot]) | (int(row[2 * slot + 1]) << 32)


def _analyse(cores, words, both_ways):
    n = len(cores)
    r = words[0]
    t0_r = _u64(r, 0)
    records = int(r[4])
    arrivals = [(_u64(words[n + 1 + k], 2) - t0_r, int(words[n + 1 + k][6])) for k in range(records)]
    per = []
    for i in range(1, n):
        b = words[i]
        rr = words[n + i]
        t0_b, rstart_seen, ping_seen, pong_sent = (_u64(b, s) for s in range(4))
        ping_sent, pong_seen = _u64(rr, 0), _u64(rr, 1)
        theta = ((ping_seen - ping_sent) + (pong_sent - pong_seen)) / 2
        rtt = (pong_seen - ping_sent) - (pong_sent - ping_seen)
        raw = t0_b - t0_r
        per.append(
            {
                "logical": cores[i],
                "phys": _phys(cores[i]),
                "raw_start_diff": raw,
                "clock_offset": theta,
                "true_start_diff": raw - theta,
                "rtt": rtt,
                "b_wait_for_r_start": (rstart_seen - t0_b) if both_ways else None,
            }
        )
    return {"t_rstart_issued": _u64(r, 1) - t0_r, "arrivals": arrivals, "per_core": per}


CASES = {f"pair_x{x}": ([REF, (x, 1)], 1, "brisc") for x in range(11) if x != REF[0]}
CASES["pair_rev_x0"] = ([(0, 1), REF], 1, "brisc")
CASES["pair_x0_ncrisc"] = ([REF, (0, 1)], 1, "ncrisc")
CASES["pair_x0_row4"] = ([REF, (0, 2)], 1, "brisc")
CASES["map_brisc"] = ([REF] + [(x, y) for y in range(10) for x in range(11) if (x, y) != REF], 0, "brisc")
CASES["map_ncrisc"] = ([REF] + [(x, y) for y in range(10) for x in range(11) if (x, y) != REF], 0, "ncrisc")


@pytest.mark.parametrize("case", list(CASES))
def test_clock_skew(case, device):
    cores, both_ways, risc = CASES[case]
    launches = _run(device, cores, both_ways, risc)
    summary = [_analyse(cores, words, both_ways) for words in launches]
    for li, s in enumerate(summary):
        if case.startswith("pair"):
            p = s["per_core"][0]
            arr = s["arrivals"][0][0] if s["arrivals"] else None
            print(
                f"\nSKEW_PAIR case={case} launch={li} R={cores[0]}{_phys(cores[0])} B={p['logical']}{p['phys']} "
                f"raw_start_diff={p['raw_start_diff']} clock_offset={p['clock_offset']:.1f} "
                f"true_start_diff={p['true_start_diff']:.1f} rtt={p['rtt']} "
                f"R_wait_for_B_start={arr} B_wait_for_R_start={p['b_wait_for_r_start']}",
                flush=True,
            )
        else:
            per = s["per_core"]
            arr = s["arrivals"]
            print(
                f"\nSKEW_MAP case={case} launch={li} arrivals_first={arr[0][0]} arrivals_last={arr[-1][0]} "
                f"n_records={len(arr)} rtt_min={min(p['rtt'] for p in per)} rtt_max={max(p['rtt'] for p in per)}",
                flush=True,
            )
            if li == LAUNCHES - 1:
                for key in ("raw_start_diff", "clock_offset", "true_start_diff"):
                    grid = {}
                    for p in per:
                        grid[p["phys"]] = p[key]
                    grid[_phys(cores[0])] = 0
                    xs = sorted({c[0] for c in grid})
                    print(f"SKEW_GRID case={case} {key} (cycles vs R, physical x across, y down)")
                    print("SKEW_GRID y\\x " + " ".join(f"{x:>6}" for x in xs))
                    for y in sorted({c[1] for c in grid}):
                        print(
                            f"SKEW_GRID {y:>3} "
                            + " ".join(f"{grid[(x, y)]:6.0f}" if (x, y) in grid else "     ." for x in xs)
                        )
                print("SKEW_ARRIVALS " + " ".join(f"{t}:{v}" for t, v in arr), flush=True)
    out = os.environ.get("SKEW_OUT")
    if out:
        path = Path(out) / f"{case}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"case": case, "cores": cores, "launches": summary}, default=str))
