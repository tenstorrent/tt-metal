"""Per run of profile_log_device.csv: the slowest core's tail (ns from the earliest kernel start).

Columns: W stream end (reader_w_stream), math end (TRISC_1 kernel), pack end (TRISC_2 kernel), writer/reader y store
start/end, core end. Also p50/p90/max over cores of (core end - W stream end) and the latest W stream end.
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
labels = sys.argv[2].split(",") if len(sys.argv) > 2 else []
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
ns = lambda c: c * 1000.0 / freq
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    runs[int(r[7])][((int(r[1]), int(r[2])), r[3], r[10])].append((int(r[5]), r[11]))
print(
    f"{'run':>5} {'label':10} {'core':8} {'w_end':>6} {'math':>6} {'pack':>6} {'wy_st':>6} {'wy_end':>6} {'ry_st':>6} {'ry_end':>6} {'end':>6} {'tail':>5} {'tail_p50':>8} {'tail_p90':>8} {'tail_max':>8} {'w_end_max':>9}"
)
for i, run in enumerate(sorted(runs)):
    span = {}
    for key, lst in runs[run].items():
        st = [t for t, ty in lst if ty == "ZONE_START"]
        en = [t for t, ty in lst if ty == "ZONE_END"]
        if st and en:
            span[key] = (min(st), max(en))
    t0 = min(v[0] for (c, r, z), v in span.items() if z.endswith("-KERNEL"))
    cores = sorted({c for (c, r, z) in span if z == "reader_w_stream"})
    get = lambda c, r, z, k: ns(span[(c, r, z)][k] - t0) if (c, r, z) in span else float("nan")
    ends = {c: max(v[1] for (cc, r, z), v in span.items() if cc == c and z.endswith("-KERNEL")) for c in cores}
    worst = max(cores, key=lambda c: ends[c])
    tails = [ns(ends[c] - span[(c, "NCRISC", "reader_w_stream")][1]) for c in cores]
    c = worst
    lab = labels[i] if i < len(labels) else ""
    print(
        f"{run:5d} {lab:10} {str(c):8} {get(c,'NCRISC','reader_w_stream',1):6.0f} {get(c,'TRISC_1','TRISC-KERNEL',1):6.0f} "
        f"{get(c,'TRISC_2','TRISC-KERNEL',1):6.0f} {get(c,'BRISC','writer_y_store',0):6.0f} {get(c,'BRISC','writer_y_store',1):6.0f} "
        f"{get(c,'NCRISC','reader_y_store',0):6.0f} {get(c,'NCRISC','reader_y_store',1):6.0f} {ns(ends[c]-t0):6.0f} "
        f"{tails[cores.index(c)]:5.0f} {statistics.median(tails):8.0f} "
        f"{sorted(tails)[int(0.9 * len(tails))]:8.0f} {max(tails):8.0f} "
        f"{max(ns(span[(cc, 'NCRISC', 'reader_w_stream')][1] - t0) for cc in cores):9.0f}"
    )
