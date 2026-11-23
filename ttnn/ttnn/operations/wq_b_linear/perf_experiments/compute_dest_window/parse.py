"""Summarize csv/<id>.csv: span (median of the last 5 of 6 dispatches) and per-RISC compute zones of the last one.
see Usage below"""
import csv, sys, statistics as st
from collections import defaultdict
from pathlib import Path

FREQ = 1.35
HERE = Path(__file__).parent


def summarize(allrows, runs):
    rows = [r for r in allrows if int(r[7]) in runs]
    spans = []
    for run in runs:
        ts = [int(r[5]) for r in rows if int(r[7]) == run and r[10].strip().endswith("-KERNEL")]
        med = sorted(ts)[len(ts) // 2]
        ts = [t for t in ts if abs(t - med) < 1e6]
        spans.append((max(ts) - min(ts)) / FREQ)
    last = [r for r in rows if int(r[7]) == runs[-1]]
    ks = [int(r[5]) for r in last if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    t0 = min(t for t in ks if abs(t - med) < 1e6)
    open_, dur, end = {}, defaultdict(dict), defaultdict(dict)
    for r in last:
        name = r[10].strip()
        if abs(int(r[5]) - med) > 1e6 or name.endswith("-FW"):
            continue
        core, risc, t = (int(r[1]), int(r[2])), r[3].strip(), int(r[5])
        k = (core, risc, name)
        if r[11].strip() == "ZONE_START":
            open_[k] = t
        elif k in open_:
            dur[(risc, name)][core] = (t - open_.pop(k)) / FREQ
            end[(risc, name)][core] = (t - t0) / FREQ
    z = {}
    for key in [("TRISC_0", "compute_matmul"), ("TRISC_1", "compute_matmul"), ("TRISC_2", "compute_matmul")]:
        if key in dur:
            z[key[0]] = (st.median(dur[key].values()), max(dur[key].values()), max(end[key].values()))
    return spans, z


# Usage: parse.py <profile_log_device.csv> <id1,id2,...>  (6 dispatches per id, in order)
allrows = list(csv.reader(open(sys.argv[1])))[2:]
ids = sys.argv[2].split(",")
runids = sorted({int(r[7]) for r in allrows})
assert len(runids) == 6 * len(ids), (len(runids), ids)
for n, i in enumerate(ids):
    spans, z = summarize(allrows, runids[6 * n : 6 * n + 6])
    zs = " ".join(f"{r}:dur_med={d:.0f}/max={m:.0f}/end_max={e:.0f}" for r, (d, m, e) in z.items())
    print(f"{i:14s} span_med={st.median(spans[1:]):7.0f} spans={[round(s) for s in spans]} {zs}")
