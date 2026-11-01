"""Per-core grid: first reader_block duration (us), number of reader_block occurrences, kernel end (us)."""
import csv, sys
from collections import defaultdict

path = "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_ = {}
first = {}
nblk = defaultdict(int)
end = {}
t0 = None
for r in rows[1:]:
    r = [c.strip() for c in r]
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    risc = r[idx["RISC processor type"]]
    z = r[idx["zone name"]]
    typ = r[idx["type"]]
    t = int(r[idx["time[cycles since reset]"]])
    k = (core, risc, z)
    if typ == "ZONE_START":
        open_[k] = t
        if z.endswith("-KERNEL"):
            t0 = t if t0 is None else min(t0, t)
    elif typ == "ZONE_END" and k in open_:
        s = open_.pop(k)
        if z == sys.argv[1] if len(sys.argv) > 1 else z == "reader_block":
            nblk[core] += 1
            if core not in first:
                first[core] = t - s
        if z == "BRISC-KERNEL":
            end[core] = t
xs = sorted({c[0] for c in end})
ys = sorted({c[1] for c in end})
print("first block us [nblocks] / end us")
for y in ys:
    print(
        f"y={y:2d} "
        + " ".join(
            f"{first.get((x,y),0)*1e-3/freq*1000:5.1f}[{nblk.get((x,y),0)}]/{(end.get((x,y),t0)-t0)/freq:4.1f}"
            if (x, y) in end
            else " " * 15
            for x in xs
        )
    )
print("x:", xs)
