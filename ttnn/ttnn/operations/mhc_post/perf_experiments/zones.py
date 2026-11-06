"""Summarize MaybeDeviceZoneScope zones from generated/profiler/.logs/profile_log_device.csv.

Per (zone, RISC): sum per core (ns), then median / max over cores; plus each RISC's kernel span and
the time from the program start to each zone's last end (critical-path position). Usage:
    python3 zones.py [profile_log_device.csv] [--core x,y]
"""
import collections
import csv
import sys

path = "generated/profiler/.logs/profile_log_device.csv"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    path = args[0]
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
ns = lambda cyc: cyc * 1000.0 / freq  # noqa: E731
starts = {}
dur = collections.defaultdict(lambda: collections.defaultdict(float))  # (zone,risc) -> core -> ns
last_end = collections.defaultdict(dict)
cnt = collections.Counter()
t0 = None
for r in rows:
    key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["zone name"].endswith("-KERNEL") or r["zone name"].startswith("mhc") or True:
        pass
    if r["type"] == "ZONE_START":
        starts[key] = t
        if r["zone name"].endswith("-FW"):
            t0 = t if t0 is None else min(t0, t)
    elif r["type"] == "ZONE_END" and key in starts:
        core = (r["core_x"], r["core_y"])
        z = (r["zone name"], r["RISC processor type"])
        dur[z][core] += ns(t - starts.pop(key))
        last_end[z][core] = t
        cnt[(core, r["RISC processor type"])] += 1
print(f"{'zone':32s} {'risc':7s} {'cores':>5s} {'med ns':>9s} {'max ns':>9s} {'max end @ns':>11s}")
order = sorted(dur, key=lambda z: (z[1], -max(dur[z].values())))
for z in order:
    v = sorted(dur[z].values())
    end = max(last_end[z].values())
    print(f"{z[0]:32s} {z[1]:7s} {len(v):5d} {v[len(v)//2]:9.0f} {v[-1]:9.0f} {ns(end - t0):11.0f}")
print("max zone-executions per (core,risc):", max(cnt.values()) if cnt else 0)
