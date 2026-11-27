# Perf 1 coordinator: per-core spatial map of one zone (or END = kernel end) for one run of profile_log_device.csv.
# usage: python3 coordinator_coremap.py <profile_log_device.csv> <run host ID> "RISC:zone,RISC:END,..."
import csv, sys
from collections import defaultdict

path, run = sys.argv[1], int(sys.argv[2])
zones = sys.argv[3].split(",")
f = open(path)
f.readline()
rows = [r for r in csv.DictReader(f, skipinitialspace=True) if int(r["run host ID"]) == run]
ev = defaultdict(list)
for r in rows:
    ev[((int(r["core_x"]), int(r["core_y"])), r["RISC processor type"], r["zone name"])].append(
        (int(r["time[cycles since reset]"]), r["type"])
    )
tot = defaultdict(int)
end = {}
start = {}
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
            end[k] = max(end.get(k, 0), t)
t0 = min(start[k] for k in start if k[2].endswith("KERNEL"))
cores = sorted({k[0] for k in tot})
xs = sorted({c[0] for c in cores})
ys = sorted({c[1] for c in cores})
for z in zones:
    risc, name = z.split(":")
    print(f"--- {name} on {risc} (us); end-of-kernel if name==END")
    print("     " + " ".join(f"x{x:<5d}" for x in xs))
    for y in ys:
        line = f"y{y:<3d} "
        for x in xs:
            if name == "END":
                k = ((x, y), risc, f"{risc.split('_')[0]}-KERNEL")
                val = (end.get(k, 0) - t0) / 1350 if k in end else None
            else:
                k = ((x, y), risc, name)
                val = tot[k] / 1350 if k in tot else None
            line += f"{val:6.1f} " if val is not None else "   -   "
        print(line)
