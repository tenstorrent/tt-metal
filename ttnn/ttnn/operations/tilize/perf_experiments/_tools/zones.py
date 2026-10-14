"""Parse generated/profiler/.logs/profile_log_device.csv for the last op run.

On this BH board the CSV 'DEVICE KERNEL DURATION' is invalid: cores sit in timer clusters with
different zero points. Spans are taken per timer cluster (cores whose KERNEL starts are within
GAP cycles of each other) and the op span = max cluster span (max end - min start of *-KERNEL).
Per-zone: per-core sum of each zone, reported as mean/max over cores. Usage:
  python3 zones.py [profile_log_device.csv] [--run N]  (N = which run-host-id, default last)
"""
import collections, csv, statistics, sys

GAP = 1_000_000
path = "generated/profiler/.logs/profile_log_device.csv"
args = [a for a in sys.argv[1:]]
run_sel = None
if "--run" in args:
    i = args.index("--run")
    run_sel = int(args[i + 1])
    del args[i : i + 2]
if args:
    path = args[0]
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
runs = sorted({int(r[7]) for r in rows})
run = runs[-1] if run_sel is None else run_sel
rows = [r for r in rows if int(r[7]) == run]
open_z = {}
zsum = collections.defaultdict(lambda: collections.defaultdict(int))
zcnt = collections.defaultdict(lambda: collections.defaultdict(int))
kern = {}
for r in rows:
    core = (int(r[1]), int(r[2]))
    risc = r[3]
    t = int(r[5])
    zone = r[10]
    typ = r[11]
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_z[key] = t
    elif typ == "ZONE_END" and key in open_z:
        d = t - open_z.pop(key)
        if zone.endswith("-KERNEL"):
            kern[(core, risc)] = (t - d, t)
        zsum[(zone, risc)][core] += d
        zcnt[(zone, risc)][core] += 1
starts = sorted(v[0] for v in kern.values())
clusters = []
for s in starts:
    if clusters and s - clusters[-1][-1] < GAP:
        clusters[-1].append(s)
    else:
        clusters.append([s])
spans = []
for c in clusters:
    lo, hi = c[0], c[-1]
    ks = [v for v in kern.values() if lo <= v[0] <= hi]
    spans.append(max(e for _, e in ks) - min(s for s, _ in ks))
span = max(spans)
ns = lambda cyc: cyc * 1000.0 / freq
print(
    f"run {run}  freq {freq} MHz  clusters {len(clusters)}  span {span} cyc = {ns(span):.0f} ns  (per-cluster ns: {[round(ns(s)) for s in spans]})"
)
for risc_k in ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"):
    for (zone, risc), per in sorted(zsum.items()):
        if zone == risc_k:
            v = list(per.values())
            print(
                f"  {zone:14s} {risc:8s} cores {len(v):3d}  p50 {ns(statistics.median(v)):8.0f}  max {ns(max(v)):8.0f} ns"
            )
for (zone, risc), per in sorted(zsum.items()):
    if zone.endswith("-FW") or zone.endswith("-KERNEL"):
        continue
    v = list(per.values())
    n = list(zcnt[(zone, risc)].values())
    print(
        f"  {zone:22s} {risc:8s} cores {len(v):3d} exec/core {statistics.mean(n):5.1f}  mean {ns(statistics.mean(v)):8.0f}  max {ns(max(v)):8.0f} ns"
    )
