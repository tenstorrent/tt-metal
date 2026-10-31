# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT early_small_wave (this dir's op copy; DRMS_REAL=1: the real op). MEASUREMENT ONLY: per-stage device-zone timeline of one distributed_rms_norm call (Refinement 3 breakdown).

    DRMS_ZONES=1 scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_zones.py -s

Runs the PERF FOCUS case (CCL_PERF_CASES like perf_measure.py) traced: spin, then ZONE_CALLS calls; the device profiler
CSV is parsed and, for the LAST call on each chip, every zone's [start, end] relative to that call's earliest marker
is summarised per (zone, RISC): first start, last end, median duration, count. Not a golden test (name keeps it out of
collection); the headline number of a variant still comes from perf_measure.py.
"""

from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("DRMS_ZONES", "1")

import pytest  # noqa: E402

from eval import ccl_perf  # noqa: E402
from eval.golden_tests.distributed_rms_norm.helpers import prepare_distributed_rms_norm  # noqa: E402
from eval.golden_tests.distributed_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.distributed_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache  # noqa: E402,F401
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if os.environ.get("DRMS_REAL", "0") not in ("", "0"):
    from ttnn.operations.distributed_rms_norm import distributed_rms_norm  # type: ignore  # noqa: E402
else:
    from early_small_wave import distributed_rms_norm  # type: ignore  # noqa: E402

ZONE_CALLS = int(os.environ.get("ZONE_CALLS", "4"))
CLOCK_GHZ = 1.35  # Blackhole AICLK (cycles -> ns)


def _selected():
    cases = _loose_cases()
    shapes = os.environ.get("DRMS_ESW_SHAPES", "")
    if shapes:  # the focus case's config with other per-chip shapes
        base = cases[0]
        out = []
        for sh in shapes.split(","):
            dims = [int(d) for d in sh.split("x")]
            shape = tuple([1] * (4 - len(dims)) + dims)
            out.append(pytest.param((shape,), base.values[1], base.values[2], id="sweep-" + "x".join(map(str, shape))))
        return out
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    if sel == "all":
        return cases
    if sel == "focus":
        return cases[:1]
    return [cases[int(i)] for i in sel.split(",")]


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):
    extras, axes = request.node.callspec.params["extras"], request.node.callspec.params["axes"]
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


def _parse(path):
    """{(chip, run host id): [(core, risc, zone, start, end)]}"""
    out = defaultdict(list)
    with open(path) as f:
        next(f)  # arch / freq line
        rd = csv.reader(f)
        hdr = [h.strip() for h in next(rd)]
        ix = {h: i for i, h in enumerate(hdr)}
        opened = {}
        for row in rd:
            row = [c.strip() for c in row]
            chip = int(row[ix["PCIe slot"]])
            run = row[ix["run host ID"]]
            core = (int(row[ix["core_x"]]), int(row[ix["core_y"]]))
            risc = row[ix["RISC processor type"]]
            name = row[ix["zone name"]]
            typ = row[ix["type"]]
            t = int(row[ix["time[cycles since reset]"]])
            key = (chip, run, core, risc, name)
            if typ == "ZONE_START":
                opened.setdefault(key, []).append(t)
            elif typ == "ZONE_END" and opened.get(key):
                out[(chip, run)].append((core, risc, name, opened[key].pop(), t))
    return out


def report(path, chips=None):
    zones = _parse(path)
    by_chip = defaultdict(list)
    for (chip, run), evs in zones.items():
        if any(n.startswith(("rd_", "rt_", "cp_", "wr_", "prt_")) for _, _, n, _, _ in evs):
            by_chip[chip].append((int(run) if run.isdigit() else 0, evs))
    for chip in sorted(by_chip):
        if chips is not None and chip not in chips:
            continue
        run, evs = max(by_chip[chip], key=lambda x: x[0])
        t0 = min(s for _, _, n, s, _ in evs if n.endswith("FW"))
        agg = defaultdict(list)
        for core, risc, name, s, e in evs:
            agg[(name, risc)].append(((s - t0) / CLOCK_GHZ, (e - t0) / CLOCK_GHZ, core))
        print(f"\n=== chip {chip} run {run}: ns from the first FW start ===")
        print(
            f"{'zone':22s} {'risc':7s} {'n':>4s} {'first_st':>8s} {'med_st':>7s} {'med_end':>7s} {'last_end':>8s} {'med_dur':>7s} {'max_dur':>7s}"
        )
        for (name, risc), v in sorted(agg.items(), key=lambda kv: min(x[0] for x in kv[1])):
            durs = [b - a for a, b, _ in v]
            print(
                f"{name[:22]:22s} {risc[:7]:7s} {len(v):4d} {min(a for a, _, _ in v):8.0f} {st.median(a for a, _, _ in v):7.0f}"
                f" {st.median(b for _, b, _ in v):7.0f} {max(b for _, b, _ in v):8.0f} {st.median(durs):7.0f} {max(durs):7.0f}"
            )


@pytest.mark.parametrize("inputs,axes,extras", _selected())
def test_perf_zones(inputs, axes, extras, perf_mesh):
    import ttnn

    call, _ = prepare_distributed_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=distributed_rms_norm, **axes
    )
    spin = ccl_perf.spin_op(perf_mesh)
    spin()
    call()
    ttnn.synchronize_device(perf_mesh)
    ttnn.ReadDeviceProfiler(perf_mesh)
    tid = ttnn.begin_trace_capture(perf_mesh, cq_id=0)
    spin()
    for _ in range(ZONE_CALLS):
        call()
    ttnn.end_trace_capture(perf_mesh, tid, cq_id=0)
    ttnn.execute_trace(perf_mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(perf_mesh)
    ttnn.ReadDeviceProfiler(perf_mesh)
    ttnn.release_trace(perf_mesh, tid)
    # the device CSV is written when the mesh closes: parse it afterwards with
    #   python tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_zones.py [csv]


if __name__ == "__main__":
    import sys

    report(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")
