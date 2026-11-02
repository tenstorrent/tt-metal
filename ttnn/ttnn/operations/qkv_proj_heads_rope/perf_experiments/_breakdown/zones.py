"""Per-zone summary of generated/profiler/.logs/profile_log_device.csv (single-op run).

Per (core, RISC, zone): summed duration and first-start / last-end relative to the earliest
*-KERNEL start on any core. Prints, per zone, the median / max across cores (us at CHIP_FREQ)."""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_ = {}
total = collections.defaultdict(float)
first = {}
last = {}
for r in rows:
    if len(r) < 12:
        continue
    core = (r[1].strip(), r[2].strip())
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_[key] = t
        first.setdefault(key, t)
    elif typ == "ZONE_END" and key in open_:
        total[key] += t - open_.pop(key)
        last[key] = t
t0 = min(v for k, v in first.items() if k[2].endswith("-KERNEL"))
us = lambda c: c / freq
by_zone = collections.defaultdict(list)
for key, d in total.items():
    by_zone[(key[1], key[2])].append((d, first[key] - t0, last[key] - t0))
print(
    f"{'risc':7} {'zone':26} {'cores':>5} {'sum_med':>8} {'sum_max':>8} {'start_med':>9} {'end_med':>8} {'end_max':>8}"
)
for (risc, zone), v in sorted(by_zone.items(), key=lambda kv: (kv[0][0], statistics.median(x[1] for x in kv[1]))):
    ds = [x[0] for x in v]
    print(
        f"{risc:7} {zone:26} {len(v):5d} {us(statistics.median(ds)):8.1f} {us(max(ds)):8.1f} "
        f"{us(statistics.median(x[1] for x in v)):9.1f} {us(statistics.median(x[2] for x in v)):8.1f} {us(max(x[2] for x in v)):8.1f}"
    )
