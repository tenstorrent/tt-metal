"""Per-stage zone summary for wq_b_linear, last dispatch. Usage: stages.py [csv]
Per (RISC, zone): summed duration per core (ns), median / max over cores, and the median / max end time (ns after op start).
Also the op span (first kernel start -> last kernel end) for every dispatch, and markers per (core, RISC) max (cap check)."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
rows = list(csv.reader(open(path)))[2:]
runs = sorted({int(r[7]) for r in rows})
spans = []
for run in runs:
    rr = [r for r in rows if int(r[7]) == run and r[10].strip().endswith("-KERNEL")]
    ts = [int(r[5]) for r in rr]
    med = sorted(ts)[len(ts) // 2]
    ts = [t for t in ts if abs(t - med) < 1e6]
    spans.append((max(ts) - min(ts)) / FREQ)
print(
    "spans ns:",
    [round(s) for s in spans],
    "median(excl first):",
    round(st.median(spans[1:] if len(spans) > 1 else spans)),
)
last = runs[-1]
rows = [r for r in rows if int(r[7]) == last]
ks = [int(r[5]) for r in rows if r[10].strip().endswith("-KERNEL")]
med = sorted(ks)[len(ks) // 2]
t0 = min(t for t in ks if abs(t - med) < 1e6)
open_ = {}
dur = defaultdict(lambda: defaultdict(float))
end = defaultdict(lambda: defaultdict(float))
cnt = defaultdict(int)
for r in rows:
    name = r[10].strip()
    if abs(int(r[5]) - med) > 1e6 or name.endswith("-FW"):
        continue
    core, risc, t = (int(r[1]), int(r[2])), r[3].strip(), int(r[5])
    cnt[(core, risc)] += 1
    k = (core, risc, name)
    if r[11].strip() == "ZONE_START":
        open_[k] = t
    elif k in open_:
        dur[(risc, name)][core] += (t - open_.pop(k)) / FREQ
        end[(risc, name)][core] = (t - t0) / FREQ
print("max markers per (core,RISC):", max(cnt.values()))
print(f"{'risc':8s} {'zone':22s} {'ncores':>6s} {'dur_med':>8s} {'dur_max':>8s} {'end_med':>8s} {'end_max':>8s}")
for risc, name in sorted(dur, key=lambda k: (k[0], st.median(end[k].values()))):
    d = list(dur[(risc, name)].values())
    e = list(end[(risc, name)].values())
    print(f"{risc:8s} {name:22s} {len(d):6d} {st.median(d):8.0f} {max(d):8.0f} {st.median(e):8.0f} {max(e):8.0f}")
