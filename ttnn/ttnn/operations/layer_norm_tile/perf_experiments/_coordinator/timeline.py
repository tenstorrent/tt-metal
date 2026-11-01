"""Zone timeline (start/end relative to the op's earliest kernel start) for the slowest core and a typical core."""
import csv
import statistics
import sys
from collections import defaultdict

path = (
    sys.argv[1] if len(sys.argv) > 1 else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
want_run = sys.argv[2] if len(sys.argv) > 2 else None
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
ev = defaultdict(list)
open_ = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run, core, risc, zone, typ = (
        r[idx["run host ID"]],
        (r[idx["core_x"]], r[idx["core_y"]]),
        r[idx["RISC processor type"]],
        r[idx["zone name"]],
        r[idx["type"]],
    )
    t = int(r[idx["time[cycles since reset]"]])
    k = (run, core, risc, zone)
    if typ == "ZONE_START":
        open_[k] = t
    elif typ == "ZONE_END" and k in open_:
        ev[run].append((core, risc, zone, open_.pop(k), t))
runs = [run for run in ev if any(e[2].startswith(("compute_", "reader_", "writer_")) for e in ev[run])]
for run in runs if want_run is None else [want_run]:
    e = ev[run]
    t0 = min(x[3] for x in e if x[2].endswith("-KERNEL"))
    span = defaultdict(int)
    for c, risc, z, s, t in e:
        if z.endswith("-KERNEL"):
            span[c] = max(span[c], t - t0)
    p50 = statistics.median(span.values())
    slow = max(span, key=span.get)
    typ = min(span, key=lambda c: abs(span[c] - p50))
    for label, core in (("SLOWEST", slow), ("TYPICAL", typ)):
        print(f"run {run} {label} core {core} end {span[core]*1000/freq:.0f} ns")
        for c, risc, z, s, t in sorted((x for x in e if x[0] == core and not x[2].endswith("-FW")), key=lambda x: x[3]):
            if risc in ("TRISC_0", "TRISC_2"):
                continue
            print(f"  {risc:8s} {z:28s} {(s-t0)*1000/freq:8.0f} -> {(t-t0)*1000/freq:8.0f}  ({(t-s)*1000/freq:6.0f})")
