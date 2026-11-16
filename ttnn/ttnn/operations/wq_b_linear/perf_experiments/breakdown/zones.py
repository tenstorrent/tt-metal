"""Per-zone summary of generated/profiler/.logs/profile_log_device.csv for the last op run.

Times are ns relative to the earliest *-KERNEL start across all cores (AICLK from the log header).
For each (RISC, zone): cores, per-core summed duration p50/max, per-core last end p50/max.
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
ev = collections.defaultdict(list)
run_ids = sorted({int(r[7]) for r in rows})
last_run = run_ids[-1]
for r in rows:
    if int(r[7]) != last_run:
        continue
    core = (int(r[1]), int(r[2]))
    ev[(core, r[3], r[10])].append((int(r[5]), r[11]))
spans = collections.defaultdict(dict)
for (core, risc, zone), lst in ev.items():
    lst.sort()
    stack, tot, first, last = [], 0, None, None
    for t, typ in lst:
        if typ == "ZONE_START":
            stack.append(t)
        elif typ == "ZONE_END" and stack:
            s = stack.pop()
            tot += t - s
            first = s if first is None else min(first, s)
            last = t if last is None else max(last, t)
    if first is not None:
        spans[(risc, zone)][core] = (tot, first, last)
t0 = min(v[1] for (risc, zone), d in spans.items() if zone.endswith("-KERNEL") for v in d.values())
ns = lambda c: c * 1000.0 / freq
print(f"run {last_run} freq {freq} MHz")
print(
    f"{'risc':7} {'zone':32} {'cores':>5} {'dur_p50':>8} {'dur_max':>8} {'start_p50':>9} {'end_p50':>8} {'end_max':>8}"
)
for (risc, zone), d in sorted(
    spans.items(), key=lambda kv: (kv[0][0], statistics.median(v[2] for v in kv[1].values()))
):
    if zone.endswith("-FW"):
        continue
    durs = [v[0] for v in d.values()]
    starts = [v[1] - t0 for v in d.values()]
    ends = [v[2] - t0 for v in d.values()]
    print(
        f"{risc:7} {zone:32} {len(d):5d} {ns(statistics.median(durs)):8.0f} {ns(max(durs)):8.0f} "
        f"{ns(statistics.median(starts)):9.0f} {ns(statistics.median(ends)):8.0f} {ns(max(ends)):8.0f}"
    )
