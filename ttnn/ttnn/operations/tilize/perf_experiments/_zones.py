# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone reader for the tilize Perf-1 breakdown.

Parses `generated/profiler/.logs/profile_log_device.csv` — the RAW marker log the
device profiler writes when `TT_METAL_DEVICE_PROFILER=1` — and folds the
`MaybeDeviceZoneScope` markers the tilize kernels carry into a per-stage table.

Two things it reports that a naive reader would miss (see
.claude/references/device-zone-scope-attribution.md §7):

  * **marker saturation** — 250 optional markers per RISC, exhaustion is SILENT.
    `saturated` flags any (core, RISC) at or near the cap.
  * **span coverage** — the fraction of the RISC's own `*-KERNEL` span that the
    user zones actually cover. Well under 1.0 means the profile is partial and
    any "dominant stage" it shows may just be the last one recorded.
"""

import csv
import os
from collections import defaultdict

LOG = "generated/profiler/.logs/profile_log_device.csv"

_FW_ZONES = {"BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL", "BRISC-FW", "NCRISC-FW", "TRISC-FW"}


def _rows(path=LOG):
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if ln.strip()]
    # The device log carries a one-line preamble before the header row.
    start = 0
    for i, ln in enumerate(lines):
        if "zone name" in ln.lower():
            start = i
            break
    reader = csv.DictReader(lines[start:], skipinitialspace=True)
    for row in reader:
        yield {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def clear(path=LOG):
    if os.path.exists(path):
        os.remove(path)


def breakdown(path=LOG):
    """-> (stages, diag). stages: name -> dict(total_ns, per_core_ns, n, risc)."""
    # (core, risc, zone) -> list of (phase, cycle, host_id). Only the LAST
    # dispatch is kept: `_measure` does a warm launch then the measured one, and
    # both land in this log.
    events = defaultdict(list)
    seq = 0
    last_host = None
    for row in _rows(path):
        h = row.get("run host ID")
        if h is not None and (last_host is None or int(h) > int(last_host)):
            last_host = h
    for row in _rows(path):
        if row.get("run host ID") != last_host:
            continue
        zone = row.get("zone name")
        if zone is None:
            continue
        risc = row.get("RISC processor type", "")
        core = (row.get("core_x"), row.get("core_y"))
        try:
            ts = int(row.get("time[cycles since reset]"))
        except (TypeError, ValueError):
            continue
        seq += 1
        # File order is the tie-break: a ZERO-LENGTH zone (a barrier that was
        # already satisfied) has identical START and END timestamps, and sorting
        # on the timestamp alone can invert them and silently DROP the pair.
        events[(core, risc, zone)].append((row.get("type"), ts, seq))

    # Wormhole/Blackhole AICLK; the profiler log is in cycles. Callers convert.
    stages = {}
    spans = {}
    markers = defaultdict(int)
    for (core, risc, zone), evs in events.items():
        markers[(core, risc)] += len(evs)
        evs.sort(key=lambda e: (e[1], e[2]))
        total = 0
        n = 0
        stack = []
        for phase, ts, _h in evs:
            if phase == "ZONE_START":
                stack.append(ts)
            elif phase == "ZONE_END" and stack:
                total += ts - stack.pop()
                n += 1
        if zone in _FW_ZONES:
            if "KERNEL" in zone:
                spans[(core, risc)] = max(spans.get((core, risc), 0), total)
            continue
        s = stages.setdefault((zone, risc), {"cycles": 0, "n": 0, "cores": set()})
        s["cycles"] += total
        s["n"] += n
        s["cores"].add(core)

    covered = defaultdict(int)
    for (core, risc, zone), evs in events.items():
        if zone in _FW_ZONES:
            continue
        evs.sort(key=lambda e: (e[1], e[2]))
        stack = []
        for phase, ts, _h in evs:
            if phase == "ZONE_START":
                stack.append(ts)
            elif phase == "ZONE_END" and stack:
                covered[(core, risc)] += ts - stack.pop()

    diag = {
        "max_markers": max(markers.values()) if markers else 0,
        "saturated": sorted(k for k, v in markers.items() if v >= 240),
        "coverage": {k: (covered.get(k, 0) / v if v else 0.0) for k, v in spans.items() if covered.get(k, 0) or v},
        "n_cores": len({c for (c, _r) in spans}),
    }
    return stages, diag


def report(stages, diag, freq_mhz=1000.0, top=None):
    lines = []
    rows = sorted(stages.items(), key=lambda kv: -kv[1]["cycles"] / max(1, len(kv[1]["cores"])))
    if top:
        rows = rows[:top]
    lines.append(f"{'stage':24s} {'risc':8s} {'cores':>6s} {'exec':>7s} {'ns/core':>10s}")
    for (name, risc), s in rows:
        per_core_cyc = s["cycles"] / max(1, len(s["cores"]))
        lines.append(f"{name:24s} {risc:8s} {len(s['cores']):6d} {s['n']:7d} {per_core_cyc / freq_mhz * 1000:10.0f}")
    lines.append(f"markers/RISC max={diag['max_markers']} (cap 250)  saturated={len(diag['saturated'])}")
    cov = diag["coverage"]
    if cov:
        by_risc = defaultdict(list)
        for (core, risc), c in cov.items():
            by_risc[risc].append(c)
        lines.append(
            "zone coverage of KERNEL span: " + ", ".join(f"{r}={sum(v)/len(v):.2f}" for r, v in sorted(by_risc.items()))
        )
    return "\n".join(lines)
