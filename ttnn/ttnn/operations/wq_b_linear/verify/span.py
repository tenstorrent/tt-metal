"""Per-dispatch device span for wq_b_linear from profile_log_device.csv (kernel zones only).

Usage: python3 span.py [csv]  -> per-dispatch span (us), median, and per-RISC end times (us after op start)
for the sender core(s) and the median worker. Cores whose timestamps are far off the median start are dropped.
"""
import csv, sys, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1350.0
rows = list(csv.reader(open(path)))[2:]
z = defaultdict(dict)  # (run, core, risc) -> {start, end}
for r in rows:
    name = r[10].strip()
    if not name.endswith("-KERNEL"):
        continue
    key = (int(r[7]), (int(r[1]), int(r[2])), r[3].strip())
    t = int(r[5])
    if r[11].strip() == "ZONE_START":
        z[key].setdefault("start", t)
    else:
        z[key]["end"] = t
runs = sorted({k[0] for k in z})
spans = []
for run in runs:
    ks = [k for k in z if k[0] == run and "start" in z[k] and "end" in z[k]]
    starts = sorted(z[k]["start"] for k in ks)
    med = starts[len(starts) // 2]
    ks = [k for k in ks if abs(z[k]["start"] - med) < 1e6]
    t0 = min(z[k]["start"] for k in ks)
    t1 = max(z[k]["end"] for k in ks)
    spans.append((t1 - t0) / FREQ * 1000)
    if run == runs[-1]:
        cores = sorted({k[1] for k in ks})
        per = defaultdict(dict)
        for k in ks:
            per[k[1]][k[2]] = (z[k]["end"] - t0) / FREQ * 1000
        print("last dispatch, end times (ns after op start):")
        senders = [c for c in cores if "TRISC_0" not in per[c]]
        for c in senders:
            print("  sender", c, {r: round(v) for r, v in sorted(per[c].items())})
        workers = [c for c in cores if "TRISC_0" in per[c]]
        for risc in ["NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "BRISC"]:
            vals = sorted(per[c][risc] for c in workers if risc in per[c])
            if vals:
                print(f"  workers {risc:8s} min {vals[0]:8.0f} med {vals[len(vals)//2]:8.0f} max {vals[-1]:8.0f}")
print("spans ns:", [round(s * 1) for s in spans])
print("median (excluding first):", round(st.median(spans[1:] if len(spans) > 1 else spans)))
