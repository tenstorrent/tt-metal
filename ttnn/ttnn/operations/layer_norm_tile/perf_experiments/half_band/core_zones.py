"""Zone timeline (us from op start) of the latest-ending core and a median core of a given class.
usage: core_zones.py <device csv> <run id> <n compute rows>"""
import csv, sys
from collections import defaultdict

path, want, nrows = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_, ev = {}, defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    if r[idx["run host ID"]] != want:
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    k = (core, r[idx["RISC processor type"]], r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        ev[core].append((k[1], k[2], open_.pop(k), t))
t0 = min(s for c in ev for (_, z, s, _) in ev[c] if z.endswith("KERNEL"))
cls = [c for c in ev if sum(1 for e in ev[c] if e[1] == "compute_add_and_stats" and e[0] == "TRISC_1") == nrows]
end = lambda c: max(e[3] for e in ev[c] if e[1].endswith("KERNEL"))
cls.sort(key=end)
for c in (cls[-1], cls[len(cls) // 2]):
    print(f"core {c} end {(end(c)-t0)/freq:.1f}")
    for risc, z, s, e in sorted(ev[c], key=lambda e: e[2]):
        if risc in ("TRISC_0", "TRISC_2") or z.endswith("KERNEL"):
            continue
        print(f"  {risc:8s} {z:28s} {(s-t0)/freq:6.1f} -> {(e-t0)/freq:6.1f}  ({(e-s)/freq:5.1f})")
