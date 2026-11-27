"""Perf 1: per-stage breakdown from the permanent MaybeDeviceZoneScope zones.

Usage: python3 stage_zones.py [csv] [--replay i]. Works on an eager profile (last run per chip) or a
trace-replay profile (probes/probe_trace_profile.py): kernel zones are split into program instances
where kernel starts jump by > 5 us (as perf/trace_gaps.py), and one instance is reported (default: the
last; instance 0 is the eager warm-up call). Per (zone, RISC): per-core occupancy in that instance
(us, mean / max over cores of all chips) and end time from the instance's start (max), plus the span
per chip. Also checks the per-(core, RISC) marker budget (250, exhausted silently).
"""
import csv, sys, bisect
from collections import defaultdict

FREQ = 1350.0
args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "generated/profiler/.logs/profile_log_device.csv"
which = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else -1
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}  # chip -> sorted instance start times
for chip, ts in kst.items():
    ts.sort()
    starts = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * FREQ:
            starts.append(b)
    inst[chip] = starts


def which_inst(chip, t):
    return bisect.bisect_right(inst[chip], t) - 1


sel = {c: (len(s) - 1 if which < 0 else which) for c, s in inst.items()}
kend = defaultdict(int)
open_t, occ, zend, markers = {}, defaultdict(float), defaultdict(int), defaultdict(int)
for r in rows:
    chip, x, y, risc, _, t = r[:6]
    t = int(t)
    i = which_inst(chip, t)
    if i != sel[chip]:
        continue
    risc = risc.strip()
    zone, typ = r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") or zone.endswith("-FW"):
        if zone.endswith("-KERNEL") and typ == "ZONE_END":
            kend[chip] = max(kend[chip], t)
        continue
    markers[(chip, x, y, risc)] += 1
    k = (chip, zone, risc, x, y)
    if typ == "ZONE_START":
        open_t[k] = t
    elif typ == "ZONE_END" and k in open_t:
        occ[k] += t - open_t.pop(k)
        zend[k] = max(zend[k], t)
chips = sorted(inst, key=int)
print("span per chip (us):", " ".join(f"{c}:{(kend[c]-inst[c][sel[c]])/FREQ:.1f}" for c in chips))
agg, aend = defaultdict(list), defaultdict(float)
for (chip, zone, risc, x, y), v in occ.items():
    agg[(zone, risc)].append(v / FREQ)
    aend[(zone, risc)] = max(aend[(zone, risc)], (zend[(chip, zone, risc, x, y)] - inst[chip][sel[chip]]) / FREQ)
print(f"{'zone':24s} {'risc':7s} {'cores':>5s} {'mean':>7s} {'max':>7s} {'end_max':>8s}")
for (zone, risc), v in sorted(agg.items(), key=lambda kv: -kv[1][0] if False else -aend[kv[0]]):
    print(f"{zone:24s} {risc:7s} {len(v):5d} {sum(v)/len(v):7.1f} {max(v):7.1f} {aend[(zone, risc)]:8.1f}")
mx = max(markers.values()) if markers else 0
print(f"max markers per (core, RISC) in this instance: {mx} (cap 250 per program run)")
