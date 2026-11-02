"""Per-core zone sums / ends (us) for chosen zones from profile_log_device.csv. Usage: percore.py zone[,zone..]"""
import collections, csv, sys

path = "generated/profiler/.logs/profile_log_device.csv"
zones = sys.argv[1].split(",")
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_, tot, last, cnt = {}, collections.defaultdict(float), {}, collections.defaultdict(int)
t0 = None
for r in rows:
    if len(r) < 12:
        continue
    core = (int(r[1]), int(r[2]))
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_[key] = t
    elif typ == "ZONE_END" and key in open_:
        tot[key] += t - open_.pop(key)
        last[key] = t
        cnt[key] += 1
for z in zones:
    print("==", z)
    items = sorted(
        ((k[0], k[1], tot[k] / freq, (last[k] - t0) / freq, cnt[k]) for k in tot if k[2] == z), key=lambda x: -x[3]
    )
    for core, risc, s, e, n in items[: int(sys.argv[2]) if len(sys.argv) > 2 else 12]:
        print(f"  {core} {risc:7} sum {s:6.1f} end {e:6.1f} n {n}")
