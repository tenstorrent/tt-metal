"""Both views of a launch's kernel time from profile_log_device.csv, median over the last N launches:
start spread (last core start - first core start), the longest per-core span (end - start on one core,
over BRISC and NCRISC: immune to per-core timer offsets) and the kernel span (last end - first start,
the DEVICE KERNEL DURATION view), in cycles, plus the per-core start offsets by physical (x, y).
Usage: spans.py <profile_log_device.csv> [N]"""
import collections
import csv
import statistics
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 11
start = collections.defaultdict(dict)
end = collections.defaultdict(dict)
with open(path) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        if r["zone name"] in ("BRISC-KERNEL", "NCRISC-KERNEL"):
            core = (int(r["core_x"]), int(r["core_y"]))
            run = int(r["run host ID"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                start[run][core] = min(t, start[run].get(core, t))
            else:
                end[run][core] = max(t, end[run].get(core, t))
runs = sorted(start)[-n:]
spread, core_max, kernel = [], [], []
offsets = collections.defaultdict(list)
for run in runs:
    t0 = min(start[run].values())
    spread.append(max(start[run].values()) - t0)
    core_max.append(max(end[run][c] - start[run][c] for c in start[run]))
    kernel.append(max(end[run].values()) - t0)
    for c, t in start[run].items():
        offsets[c].append(t - t0)
cores = sorted(offsets, key=lambda c: (c[1], c[0]))
print("start offset by phys core (cyc): " + " ".join(f"{c}:{statistics.median(offsets[c]):.0f}" for c in cores))
print(
    f"spread_cyc {statistics.median(spread):.0f} core_span_max_cyc {statistics.median(core_max):.0f} "
    f"kernel_cyc {statistics.median(kernel):.0f} launches {len(runs)} cores {len(cores)}"
)
