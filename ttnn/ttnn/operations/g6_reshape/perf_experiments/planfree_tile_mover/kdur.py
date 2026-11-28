"""kdur.py <profile_log_device.csv>: per-run device kernel duration (first *-KERNEL start to last *-KERNEL end over
all cores and RISCs), median over runs 2..N, in ns at 1.35 GHz."""
import csv, sys, statistics, collections

f = open(sys.argv[1])
next(f)
st = collections.defaultdict(lambda: [1 << 62, 0])
for r in csv.DictReader(f, skipinitialspace=True):
    if not r["zone name"].endswith("-KERNEL"):
        continue
    t = int(r["time[cycles since reset]"])
    k = int(r["run host ID"])
    if r["type"] == "ZONE_START":
        st[k][0] = min(st[k][0], t)
    else:
        st[k][1] = max(st[k][1], t)
runs = sorted(st)
d = [(st[k][1] - st[k][0]) / 1.35 for k in runs[1:]]
print(f"KDUR n={len(d)} median={statistics.median(d):.0f} min={min(d):.0f} max={max(d):.0f}")
