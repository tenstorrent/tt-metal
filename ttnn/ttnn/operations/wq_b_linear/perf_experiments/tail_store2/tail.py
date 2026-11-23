"""tail_store2 analysis. Usage: tail.py csv
Per dispatch (2..N): span; per worker tail = last x_wait_rest end -> y_barrier end (median, max over workers);
also worker x_wait_rest / TRISC_2 / y_wait / y_issue / y_barrier end (median, max). Prints median over dispatches."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
runs = sorted({int(r[7]) for r in rows})
out = []
for run in runs:
    rr = [r for r in rows if int(r[7]) == run]
    ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    ks = [t for t in ks if abs(t - med) < 1e6]
    t0 = min(ks)
    span = (max(ks) - t0) / FREQ
    end = defaultdict(dict)
    for r in rr:
        t = int(r[5])
        name = r[10].strip()
        if abs(t - med) > 1e6 or r[11].strip() != "ZONE_END" or name.endswith("-FW"):
            continue
        key = r[3].strip() + "-K" if name.endswith("-KERNEL") else name
        core = (int(r[1]), int(r[2]))
        end[key][core] = max(end[key].get(core, 0), (t - t0) / FREQ)
    workers = list(end["y_barrier"].keys())
    tails = [end["y_barrier"][c] - end["x_wait_rest"][c] for c in workers if c in end["x_wait_rest"]]
    d = {"span": span, "tail_med": st.median(tails), "tail_max": max(tails)}
    tw = [end["y_barrier"][c] - end["NCRISC-K"][c] for c in workers if c in end["NCRISC-K"]]
    d["tailW_med"], d["tailW_max"] = st.median(tw), max(tw)
    for zn in ["y_issue", "y_barrier"]:
        pass
    for k in ["x_wait_rest", "TRISC_2-K", "y_wait", "y_issue", "y_barrier", "NCRISC-K"]:
        v = [end[k][c] for c in workers if c in end[k]]
        if v:
            d[k + "_med"] = st.median(v)
            d[k + "_max"] = max(v)
    out.append(d)
use = out[1:] if len(out) > 1 else out
print("spans:", [round(d["span"]) for d in out])
for k in use[0]:
    print(f"  {k:18s} {st.median(d[k] for d in use):8.0f}")
