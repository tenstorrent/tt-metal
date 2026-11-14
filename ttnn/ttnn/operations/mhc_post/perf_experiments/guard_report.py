"""Perf 1 guard-set report: python guard_report.py <ops_perf_results.csv> [<other.csv> ...] -> median DEVICE KERNEL
DURATION (us) per guard case (guard_set.py order, REPS calls each), one column per CSV."""
import csv, statistics as st, sys

CASES = [
    "plain fp32 T160 C5120",
    "plain fp32 T1024 C1792",
    "plain T160",
    "stats T160",
    "collapse T160",
    "collapse+norm T160",
    "plain T1280",
    "stats T1280",
    "collapse+norm T1280",
    "collapse+norm sep T1280",
]
REPS = 4
cols = []
for path in sys.argv[1:]:
    d = [
        int(x["DEVICE KERNEL DURATION [ns]"]) / 1000
        for x in csv.DictReader(open(path))
        if x["OP CODE"] == "GenericOpDeviceOperation"
    ]
    cols.append([d[i * REPS : (i + 1) * REPS] for i in range(len(CASES))])
print(f"{'case':26s}" + "".join(f"{p.split('/')[-1][:22]:>24s}" for p in sys.argv[1:]))
for i, c in enumerate(CASES):
    print(f"{c:26s}" + "".join(f"{st.median(col[i]):10.1f} ({min(col[i]):6.1f}-{max(col[i]):6.1f})" for col in cols))
