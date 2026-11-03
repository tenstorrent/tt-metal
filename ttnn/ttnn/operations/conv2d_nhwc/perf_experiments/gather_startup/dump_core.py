"""Dump one core's zone events up to a cycle limit. Usage: dump_core.py csv x y [limit]"""
import csv, sys

path, cx, cy = sys.argv[1:4]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else 25000
with open(path) as fh:
    fh.readline()
    rows = list(csv.reader(fh))
cols = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(cols)}
ev = [r for r in rows[1:] if len(r) >= len(cols) and r[ix["core_x"]].strip() == cx and r[ix["core_y"]].strip() == cy]
last = max(int(r[ix["run host ID"]]) for r in ev)
ev = [
    (
        int(r[ix["time[cycles since reset]"]]),
        r[ix["RISC processor type"]].strip(),
        r[ix["zone name"]].strip(),
        r[ix["type"]].strip(),
    )
    for r in ev
    if int(r[ix["run host ID"]]) == last
]
ev.sort()
t0 = min(t for t, _, n, ty in ev if n.endswith("KERNEL") and ty == "ZONE_START")
st = {}
for t, risc, n, ty in ev:
    if t - t0 > lim:
        break
    if ty == "ZONE_START":
        st[(risc, n)] = t
    elif ty == "ZONE_END" and (risc, n) in st and risc in ("NCRISC", "BRISC", "TRISC_2"):
        print(f"{risc:7s} {n:26s} {st[(risc,n)]-t0:7d} -> {t-t0:7d} ({t-st[(risc,n)]})")
