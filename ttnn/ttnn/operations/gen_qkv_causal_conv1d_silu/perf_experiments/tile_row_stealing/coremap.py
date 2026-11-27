# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core grid maps of one run of a profile_log_device.csv (host-side only; tile_row_stealing).

    python3 coremap.py <csv> <run host ID> "RISC:zone,RISC:END,RISC:EXEC:zone,TS:steal_blocks,..."
  RISC:zone        summed duration of the zone per core (us)
  RISC:END         end of the RISC's *-KERNEL zone, us from the program's first kernel start
  RISC:EXEC:zone   executions of the zone per core (e.g. NCRISC:EXEC:reader_rows_issue = tile-rows loaded)
  TS:name          the core's TS_DATA value (e.g. steal_blocks = blocks this core stole)
Grid: physical x columns, physical y rows.
"""
import csv
import sys
from collections import defaultdict

path, run = sys.argv[1], int(sys.argv[2])
specs = sys.argv[3].split(",")
f = open(path)
header = f.readline()
mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) if "CHIP_FREQ" in header else 1350.0
rows = [r for r in csv.DictReader(f, skipinitialspace=True) if int(r["run host ID"]) == run]
ev = defaultdict(list)
ts = defaultdict(dict)
for r in rows:
    c = (int(r["core_x"]), int(r["core_y"]))
    if r["type"] == "TS_DATA":
        ts[r["zone name"]][c] = int(r["data"])
        continue
    ev[(c, r["RISC processor type"], r["zone name"])].append((int(r["time[cycles since reset]"]), r["type"]))
tot, cnt, end, start = defaultdict(int), defaultdict(int), {}, {}
for k, v in ev.items():
    v.sort()
    st = []
    for t, ty in v:
        if ty == "ZONE_START":
            st.append(t)
            start[k] = min(start.get(k, t), t)
        elif st:
            s = st.pop()
            tot[k] += t - s
            cnt[k] += 1
            end[k] = max(end.get(k, 0), t)
t0 = min(start[k] for k in start if k[2].endswith("KERNEL"))
cores = sorted({k[0] for k in tot})
xs = sorted({c[0] for c in cores})
ys = sorted({c[1] for c in cores})
for spec in specs:
    parts = spec.split(":")
    val = {}
    if parts[0] == "TS":
        val = ts.get(parts[1], {})
        title = f"TS_DATA {parts[1]}"
    elif parts[1] == "END":
        risc = parts[0]
        val = {
            c: (end[(c, risc, f"{risc.split('_')[0]}-KERNEL")] - t0) / mhz
            for c in cores
            if (c, risc, f"{risc.split('_')[0]}-KERNEL") in end
        }
        title = f"{risc} kernel end (us)"
    elif parts[1] == "EXEC":
        risc, z = parts[0], parts[2]
        val = {c: cnt[(c, risc, z)] for c in cores if (c, risc, z) in cnt}
        title = f"{risc} {z} executions"
    else:
        risc, z = parts
        val = {c: tot[(c, risc, z)] / mhz for c in cores if (c, risc, z) in tot}
        title = f"{risc} {z} sum (us)"
    print(f"--- run {run}: {title}")
    print("  y\\x " + "".join(f"{x:7d}" for x in xs))
    for y in ys:
        line = f" {y:4d} "
        for x in xs:
            v = val.get((x, y))
            line += "      -" if v is None else (f" {v:6.1f}" if isinstance(v, float) else f" {v:6d}")
        print(line)
