# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage table of the last q_stem call per chip from the permanent MaybeDeviceZoneScope zones (zone_probe.py).
For every (zone, RISC): cores, start / end (us from the chip's earliest kernel start; min / med / max) and duration
(med / max). Also checks the per-RISC marker budget (a RISC near 250 optional markers has silently lost zones).
Host-side script, no device.

    python3 tests/ttnn/unit_tests/operations/q_stem/zone_stages.py [profile_log_device.csv]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))


def q(v):
    v = sorted(v)
    return v[0], v[len(v) // 2], v[-1]


for chip in sorted(last):
    st = defaultdict(dict)  # (core, risc, zone) -> start/end
    markers = defaultdict(int)
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        key = (c, r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        d = st[key]
        if r["type"] == "ZONE_START":
            d["s"] = min(d.get("s", t), t)
        elif r["type"] == "ZONE_END":
            d["e"] = max(d.get("e", t), t)
        if not r["zone name"].endswith("KERNEL"):
            markers[(c, r["RISC processor type"])] += 1
    t0 = min(d["s"] for k, d in st.items() if k[2].endswith("KERNEL") and "s" in d)
    end = max(d["e"] for k, d in st.items() if k[2].endswith("KERNEL") and "e" in d)
    print(
        f"chip {chip}: program end {(end - t0) / MHZ:.1f} us; max user markers per RISC {max(markers.values(), default=0)}"
    )
    agg = defaultdict(list)
    for (c, risc, z), d in st.items():
        if "s" in d and "e" in d:
            agg[(z, risc)].append(((d["s"] - t0) / MHZ, (d["e"] - t0) / MHZ))
    for (z, risc), v in sorted(agg.items(), key=lambda kv: q([x[0] for x in kv[1]])[1]):
        s, e, du = q([x[0] for x in v]), q([x[1] for x in v]), q([x[1] - x[0] for x in v])
        print(
            f"  {z:22s} {risc:8s} n={len(v):3d} start {s[0]:6.1f}/{s[1]:6.1f}/{s[2]:6.1f}"
            f"  end {e[0]:6.1f}/{e[1]:6.1f}/{e[2]:6.1f}  dur med {du[1]:6.1f} max {du[2]:6.1f}"
        )
