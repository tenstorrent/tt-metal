"""Per core type (rotary = has an NCRISC reader_table zone, v = not): median / max of selected zones'
start / end / duration in us relative to the earliest *-KERNEL start."""
import collections, csv, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
ZONES = [
    "compute_matmul",
    "compute_last_k_epilogue",
    "writer_store",
    "reader_store",
    "BRISC-KERNEL",
    "NCRISC-KERNEL",
    "TRISC-KERNEL",
]
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in list(csv.reader(f))[1:] if len(r) >= 12]
# Last op run (run host ID) that has the compute tail zone.
run = [r[7].strip() for r in rows if r[10].strip() == "compute_last_k_epilogue"][-1]
rows = [r for r in rows if r[7].strip() == run]
open_, span = {}, {}
rotary = set()
for r in rows:
    if len(r) < 12:
        continue
    core = (r[1].strip(), r[2].strip())
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone == "reader_table":
        rotary.add(core)
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_[key] = t
        span.setdefault(key, [t, t])
    elif typ == "ZONE_END" and key in open_:
        span[key][1] = t
t0 = min(v[0] for k, v in span.items() if k[2].endswith("-KERNEL"))
for kind in ("rotary", "v"):
    print(f"[{kind}]")
    for zone in ZONES:
        vals = [
            (v[0] - t0, v[1] - t0)
            for k, v in span.items()
            if k[2] == zone
            and (k[1] in ("TRISC_1",) or not k[1].startswith("TRISC"))
            and ((k[0] in rotary) == (kind == "rotary"))
        ]
        if not vals:
            continue
        s = [a / freq for a, _ in vals]
        e = [b / freq for _, b in vals]
        d = [(b - a) / freq for a, b in vals]
        print(
            f"  {zone:26} n={len(vals):3d} start_med={statistics.median(s):6.1f} end_med={statistics.median(e):6.1f} end_max={max(e):6.1f} span_med={statistics.median(d):6.1f}"
        )
