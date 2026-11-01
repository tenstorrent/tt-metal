"""Per-stage zone summary from profile_log_device.csv (one run host ID per op).

For each run host ID that contains layer_norm zones: per (RISC, zone) the per-core summed duration,
reported as max / p50 over cores, plus the per-core *-KERNEL span and the marker-count check.
Cores are grouped by their KERNEL span so the critical (slowest) cores are visible.
"""
import csv
import statistics
import sys
from collections import defaultdict

path = (
    sys.argv[1] if len(sys.argv) > 1 else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]
idx = {c: i for i, c in enumerate(cols)}
starts = {}
dur = defaultdict(lambda: defaultdict(float))  # run -> (core,risc,zone) -> ns
count = defaultdict(int)
markers = defaultdict(int)
last_user_end = defaultdict(int)
kernel_span = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    core = (r[idx["core_x"]], r[idx["core_y"]])
    risc = r[idx["RISC processor type"]]
    zone = r[idx["zone name"]]
    typ = r[idx["type"]]
    t = int(r[idx["time[cycles since reset]"]])
    key = (run, core, risc, zone)
    markers[(run, core, risc)] += 1
    if typ == "ZONE_START":
        starts[key] = t
    elif typ == "ZONE_END" and key in starts:
        d = (t - starts.pop(key)) * 1000.0 / freq
        dur[run][(core, risc, zone)] += d
        count[(run, core, risc, zone)] += 1
        if zone.endswith("-KERNEL"):
            kernel_span[(run, core, risc)] = (starts.get((run, core, risc, zone)), t)
runs = [run for run in dur if any(k[2].startswith(("compute_", "reader_", "writer_")) for k in dur[run])]
for run in runs:
    d = dur[run]
    cores = sorted({k[0] for k in d})
    # critical cores: top TRISC-KERNEL span
    kspan = {
        c: max(d.get((c, rk, rk + "-KERNEL"), 0) for rk in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"))
        for c in cores
    }
    ordered = sorted(cores, key=lambda c: kspan[c])
    p50 = statistics.median(kspan.values())
    crit = [c for c in cores if kspan[c] > 1.2 * p50] or ordered[-max(1, len(ordered) // 10) :]
    rest = [c for c in cores if c not in crit]
    print(
        f"run {run}: {len(cores)} cores, kernel span max {max(kspan.values()):.0f} p50 {p50:.0f} ns; critical cores {len(crit)}"
    )
    zones = sorted({(k[1], k[2]) for k in d})
    print(f"{'risc':8s} {'zone':28s} {'crit_avg':>9s} {'crit_max':>9s} {'rest_avg':>9s} {'n/core':>6s}")
    for risc, zone in zones:
        cv = [d.get((c, risc, zone), 0) for c in crit]
        rv = [d.get((c, risc, zone), 0) for c in rest] or [0]
        n = max(count[(run, c, risc, zone)] for c in crit)
        print(f"{risc:8s} {zone:28s} {statistics.mean(cv):9.0f} {max(cv):9.0f} {statistics.mean(rv):9.0f} {n:6d}")
    mk = max(v for k, v in markers.items() if k[0] == run)
    print(f"max markers per (core,risc): {mk}")
