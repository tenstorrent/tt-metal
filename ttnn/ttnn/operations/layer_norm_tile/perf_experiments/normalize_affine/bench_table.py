"""Pair the first N NA_BENCH lines (profiled pass) with the newest ops_perf CSV rows; print per-rep ns."""
import glob, os, sys

log = sys.argv[1]
csv = (
    sys.argv[2]
    if len(sys.argv) > 2
    else max(
        glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"),
        key=os.path.getmtime,
    )
)
durs = [int(l.split(",")[19]) for l in open(csv).read().splitlines()[1:]]
lines = [l.split() for l in open(log) if l.startswith("NA_BENCH")][: len(durs)]
d = {}
for p, dur in zip(lines, durs):
    kv = dict(x.split("=") for x in p[1:])
    key = (kv["rows"], kv["Wt"], kv["fp32_dest"] + "/" + kv.get("fid", "HiFi4"))
    d.setdefault(key, {}).setdefault(kv["variant"], {})[int(kv["reps"])] = dur
for key, vs in d.items():
    base = None
    for v, r in vs.items():
        per = (r[33] - r[1]) / 32
        base = per if v == "baseline" else base
        print(
            f"R={key[0]} Wt={key[1]:3s} fp32={key[2]:12s} {v:12s} reps33={r[33]:7d} per_rep={per:8.1f} ns  per_tile={per/int(key[0])/int(key[1]):6.1f}  vs_base={per/base if base else 0:5.3f}"
        )
