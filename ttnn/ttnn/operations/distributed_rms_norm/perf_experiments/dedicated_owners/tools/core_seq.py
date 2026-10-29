import csv, sys
from collections import defaultdict

log = sys.argv[5]
chip = int(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
tmin = float(sys.argv[4])
with open(log) as fh:
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
rr = [r for r in rows if int(r["PCIe slot"]) == chip]
rid = max(int(r["run host ID"]) for r in rr)
rr = [r for r in rr if int(r["run host ID"]) == rid]
stack = defaultdict(list)
out = []
t0 = None
for r in rr:
    t = int(r["time[cycles since reset]"])
    if r["zone name"].endswith("-KERNEL") and r["type"] == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
for r in rr:
    if int(r["core_x"]) != x or int(r["core_y"]) != y:
        continue
    key = (r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        stack[key].append(t)
    elif r["type"] == "ZONE_END" and stack[key]:
        s = stack[key].pop()
        out.append(((s - t0) * 1000 / mhz, (t - t0) * 1000 / mhz, key))
for s, e, k in sorted(out):
    if e / 1000 >= tmin and not k[1].endswith("-FW"):
        print(f"{s:8.0f} {e:8.0f} {e-s:6.0f} {k[0]:8s} {k[1]}")
