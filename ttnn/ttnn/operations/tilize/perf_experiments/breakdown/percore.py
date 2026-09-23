"""Per-core start/end of each RISC-V's *-KERNEL zone, relative to the earliest kernel start (cycles)."""
import csv, glob, os, sys
from collections import defaultdict

root = os.path.join(os.path.dirname(__file__), "../../../../../..")
rep = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(root, "generated/profiler/reports/*/")))[-1]
with open(os.path.join(rep, "profile_log_device.csv")) as f:
    next(f)
    rd = csv.reader(f)
    hdr = [h.strip() for h in next(rd)]
    data = [dict(zip(hdr, [c.strip() for c in r])) for r in rd]
last = max({d["run host ID"] for d in data}, key=int)
st, en = {}, {}
for d in data:
    if d["run host ID"] != last or not d["zone name"].endswith("-KERNEL"):
        continue
    k = (int(d["core_x"]), int(d["core_y"]), d["RISC processor type"])
    (st if d["type"] == "ZONE_START" else en)[k] = int(d["time[cycles since reset]"])
t0 = min(st.values())
risc = sys.argv[2] if len(sys.argv) > 2 else "BRISC"
grid = defaultdict(dict)
for (x, y, r), s in st.items():
    if r == risc:
        grid[y][x] = (s - t0, en[(x, y, r)] - t0)
xs = sorted({x for y in grid for x in grid[y]})
print(f"{risc}: start/end (cycles from first kernel start); rows = core_y (NoC0 coords), cols = core_x")
print("y\\x " + " ".join(f"{x:>11d}" for x in xs))
for y in sorted(grid):
    print(f"{y:3d} " + " ".join(f"{grid[y][x][0]:5d}/{grid[y][x][1]:5d}" if x in grid[y] else " " * 11 for x in xs))
