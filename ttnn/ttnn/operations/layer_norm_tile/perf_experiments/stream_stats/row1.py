import csv, sys, statistics
from collections import defaultdict

path, want = sys.argv[1], sys.argv[2]
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_ = {}
ev = defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    if r[idx["run host ID"]] != want or r[idx["RISC processor type"]] != "TRISC_1":
        continue
    core = (r[idx["core_x"]], r[idx["core_y"]])
    z = r[idx["zone name"]]
    typ = r[idx["type"]]
    t = int(r[idx["time[cycles since reset]"]])
    k = (core, z)
    if typ == "ZONE_START":
        open_[k] = t
    elif typ == "ZONE_END" and k in open_:
        ev[core].append((z, open_.pop(k), t))
dur = defaultdict(list)
for c, e in ev.items():
    e.sort(key=lambda x: x[1])
    stats = [x for x in e if x[0] == "compute_add_and_stats"]
    if len(stats) != 2:
        continue
    # row-1 phases: everything after the second add_and_stats start
    start = stats[1][1]
    for z, s, t in e:
        if s >= start and not z.endswith("KERNEL"):
            dur[z].append((t - s) / freq * 1000)
for z, v in dur.items():
    print(f"  {z:28s} {statistics.median(v):6.0f} ns")
print("  total row1", sum(statistics.median(v) for z, v in dur.items()))
