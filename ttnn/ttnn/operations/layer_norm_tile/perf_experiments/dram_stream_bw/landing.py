"""landing.py [device csv|-] [run id|-] [zone]: per-core grid of the first <zone> end (default writer_residual_block) (us) and BRISC end."""
import csv, sys
from collections import defaultdict

path = (
    sys.argv[1]
    if len(sys.argv) > 1 and sys.argv[1] != "-"
    else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
want = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
zone = sys.argv[3] if len(sys.argv) > 3 else "writer_residual_block"
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_, ev = {}, defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if want and run != want:
        continue
    k = (run, int(r[idx["core_x"]]), int(r[idx["core_y"]]), r[idx["RISC processor type"]], r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        ev[run].append((k[1], k[2], k[3], k[4], open_.pop(k), t))
for run, e in ev.items():
    if not any(z == zone for *_, z, _, _ in e):
        continue
    t0 = min(s for *_, z, s, _ in e if z.endswith("-KERNEL"))
    first, end = {}, {}
    for x, y, risc, z, s, t in e:
        if z == zone and (x, y) not in first:
            first[(x, y)] = (t - t0) / freq
        if z == "BRISC-KERNEL":
            end[(x, y)] = (t - t0) / freq
    xs = sorted({c[0] for c in end})
    ys = sorted({c[1] for c in end})
    for label, d in ((f"first {zone} end", first), ("BRISC end", end)):
        print(f"run {run}: {label} (us), rows=y cols=x {xs}")
        for y in ys:
            print(f"  y={y:2d} " + " ".join(f"{d[(x, y)]:5.1f}" if (x, y) in d else "    ." for x in xs))
