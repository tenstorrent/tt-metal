"""Per-run, per-(RISC, zone) mean duration (ns) for the single-core stats bench."""
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
    for (rn, risc, zone), v in sorted(acc.items()):
        if (
            rn == run
            and (zone.startswith(("bench_", "compute_", "diag_")) or zone.endswith("KERNEL"))
            and risc.startswith("TRISC")
        ):
            print(f"  {risc:8s} {zone:28s} n={len(v):3d} mean={sum(v)/len(v):9.0f} ns")
