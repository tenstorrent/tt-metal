# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone timeline for ONE variant of the `fold_gather_overlap` bake-off.

Prints (a) the ROOT core's zone timeline in ns from the earliest BRISC-FW marker,
and (b) the spread of every member's `writer_gather_ship` END -- i.e. WHEN the
partials actually land, which is the whole premise of the idea.

    RMS_BENCH_VARIANTS=<one label>  RMS_BENCH_CASES=<one case>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_STAGE_ZONES"] = "1"

import collections
import csv
import statistics
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm_ttnn.perf_experiments.fold_gather_overlap.test_bench_fold_overlap import (
    CASES,
    VARIANTS,
    build,
    pcc,
    relrms,
    _read_kernel_ns,
)
from ttnn.operations.rms_norm_ttnn.perf_experiments.fold_gather_overlap import (
    rms_norm_ttnn_program_descriptor as PD,
)

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35  # cycles -> ns


def _rows():
    rows = []
    with CSVP.open() as fh:
        fh.readline()
        rdr = csv.reader(fh)
        header = [h.strip() for h in next(rdr)]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) < len(header):
                continue
            rows.append(r)
    return rows, idx


def test_zones():
    label = os.environ.get("RMS_BENCH_VARIANTS", "base").split(",")[0]
    name = os.environ.get("RMS_BENCH_CASES", "G1").split(",")[0]
    assert label in VARIANTS and name in CASES
    if CSVP.exists():
        CSVP.unlink()
    knobs = VARIANTS[label]
    saved = {k: getattr(PD, k) for k in knobs}
    for k, v in knobs.items():
        setattr(PD, k, v)
    device = ttnn.open_device(device_id=0)
    try:
        run, expected, live = build(device, name)
        out = run()
        got = ttnn.to_torch(out)
        print(f"ZONE case={name} variant={label} pcc={pcc(got, expected):.7f} relrms={relrms(got, expected):.6f}")
        del out, got
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)
        run()
        ttnn.synchronize_device(device)
        print(
            f"ZONE case={name} variant={label} ns={_read_kernel_ns(device)} "
            f"fold_runs={PD.LAST_FOLD_RUNS} stage_zones={PD.STAGE_ZONES}"
        )
    finally:
        ttnn.close_device(device)
        for k, v in saved.items():
            setattr(PD, k, v)

    import shutil

    if CSVP.exists():
        shutil.copy(CSVP, "/tmp/fgo_zone_capture.csv")
    rows, idx = _rows()
    print(f"ZONE csv_rows={len(rows)} cols={sorted(idx)}")
    ci = {
        k: idx[k]
        for k in (
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "zone name",
            "type",
            "run host ID",
        )
    }
    allruns = sorted({int(r[ci["run host ID"]]) for r in rows})
    znames = collections.Counter(r[ci["zone name"]].strip() for r in rows)
    print(f"ZONE runs={allruns[-5:]} nzones={len(znames)}")
    print("ZONE zone_names=" + ", ".join(f"{k}:{v}" for k, v in znames.most_common(50)))
    target = allruns[-1]
    rows = [r for r in rows if int(r[ci["run host ID"]]) == target]
    t0 = min(int(r[ci["time[cycles since reset]"]]) for r in rows)

    # every (core, risc, zone) start/end pair
    stack = collections.defaultdict(list)
    events = []  # (core, risc, zone, start_ns, end_ns)
    for r in rows:
        core = (int(r[ci["core_x"]]), int(r[ci["core_y"]]))
        risc = r[ci["RISC processor type"]].strip()
        zone = r[ci["zone name"]].strip()
        typ = r[ci["type"]].strip()
        t = (int(r[ci["time[cycles since reset]"]]) - t0) / FREQ
        if typ == "ZONE_START":
            stack[(core, risc, zone)].append(t)
        elif typ == "ZONE_END" and stack[(core, risc, zone)]:
            events.append((core, risc, zone, stack[(core, risc, zone)].pop(), t))

    # the ROOT is the core whose BRISC ran `writer_gather_wait`
    roots = {c for c, risc, z, s, e in events if z == "writer_gather_wait"}
    print(f"ZONE roots={sorted(roots)}")
    ships = sorted(e for c, risc, z, s, e in events if z == "writer_gather_ship")
    if ships:
        print(
            f"ZONE gather_ship END across cores: n={len(ships)} min={ships[0]:.0f} "
            f"med={statistics.median(ships):.0f} max={ships[-1]:.0f} spread={ships[-1] - ships[0]:.0f}"
        )
        print("ZONE gather_ship END sorted: " + " ".join(f"{v:.0f}" for v in ships))
        per_core = sorted(
            ((c, e) for c, risc, z, s_, e in events if z == "writer_gather_ship"),
            key=lambda t: (t[0][1], t[0][0]),
        )
        print("ZONE gather_ship END by core (row-major, == slot order): ")
        for c, e in per_core:
            print(f"ZONE   core={c} ship_end={e:.0f}")
    for root in sorted(roots):
        print(f"ZONE ---- root core {root} timeline (ns from earliest marker) ----")
        for c, risc, z, s, e in sorted((x for x in events if x[0] == root), key=lambda x: x[3]):
            print(f"ZONE   {s:8.0f} - {e:8.0f}  {e - s:7.0f}  {risc:8s} {z}")
