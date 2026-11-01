"""Per-run, per-(TRISC, zone) mean duration (ns) for the single-core chain bench, skipping each zone's first
execution (iteration 0 waits on the reader). Usage: python3 zones.py [profile_log_device.csv]"""
import csv, sys
from collections import defaultdict

path = (
    sys.argv[1] if len(sys.argv) > 1 else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
starts, acc = {}, defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    key = (r[idx["run host ID"]], r[idx["RISC processor type"]], r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        starts[key] = t
    elif key in starts:
        acc[key].append((t - starts.pop(key)) * 1000.0 / freq)
for run in sorted({k[0] for k in acc}, key=int):
    print(f"run {run}")
    order = ["TRISC_0", "TRISC_1", "TRISC_2"]
    zones = sorted(
        {z for (rn, _, z) in acc if rn == run and (z.startswith(("compute_", "diag_")) or z.endswith("KERNEL"))}
    )
    print(f"  {'zone':24s} " + " ".join(f"{o:>9s}" for o in order))
    for z in zones:
        vals = []
        for o in order:
            v = acc.get((run, o, z), [])
            v = v[1:] if len(v) > 1 else v
            vals.append(f"{sum(v)/len(v):9.0f}" if v else f"{'-':>9s}")
        print(f"  {z:24s} " + " ".join(vals))
