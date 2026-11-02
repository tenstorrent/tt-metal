"""Per-launch median zone duration (us) across cores for one RISC, from a multi-op profile_log_device.csv.
Usage: zones_seq.py <csv> [risc=TRISC_1] [zones=compute_matmul,compute_last_k_epilogue]"""
import collections
import csv
import statistics
import sys

path = sys.argv[1]
risc_sel = sys.argv[2] if len(sys.argv) > 2 else "TRISC_1"
zones = (sys.argv[3] if len(sys.argv) > 3 else "compute_matmul,compute_last_k_epilogue").split(",")
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_ = {}
seq = collections.defaultdict(list)  # (core, zone) -> [durations in launch order]
for r in rows:
    if len(r) < 12 or r[3].strip() != risc_sel:
        continue
    core, t, zone, typ = (r[1].strip(), r[2].strip()), int(r[5]), r[10].strip(), r[11].strip()
    if zone not in zones:
        continue
    if typ == "ZONE_START":
        open_[(core, zone)] = t
    elif typ == "ZONE_END" and (core, zone) in open_:
        seq[(core, zone)].append(t - open_.pop((core, zone)))
for zone in zones:
    per_launch = collections.defaultdict(list)
    for (core, z), ds in seq.items():
        if z == zone:
            for i, d in enumerate(ds):
                per_launch[i].append(d)
    for i in sorted(per_launch):
        cyc = statistics.median(per_launch[i])
        print(f"ZONE launch={i} {risc_sel} {zone} med_us={cyc / freq:.2f} med_cycles={cyc:.0f}")
