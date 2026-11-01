"""Per-row stage timing on the 2-row (critical) cores from profile_log_device.csv.

For each run and each TRISC, prints the median over 2-row cores of each zone's duration for row 0 and
row 1, plus the span 'add->mean' = start of the first of {compute_residual_add, compute_row_mean,
compute_wait_z} to the end of compute_row_mean (row's z+mean stage incl. any packer lag) and
'add->center' = same start to the end of compute_center.
"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]
idx = {c: i for i, c in enumerate(cols)}
ev = defaultdict(list)  # (run, core, risc, zone) -> [(start, end)]
open_ = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    key = (
        r[idx["run host ID"]],
        (r[idx["core_x"]], r[idx["core_y"]]),
        r[idx["RISC processor type"]],
        r[idx["zone name"]],
    )
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[key] = t
    elif key in open_:
        ev[key].append((open_.pop(key), t))
ns = lambda c: c * 1000.0 / freq
runs = sorted({k[0] for k in ev}, key=int)
STAGES = ["compute_residual_add", "compute_wait_z", "compute_row_mean", "compute_center"]
for run in runs:
    cores = {k[1] for k in ev if k[0] == run and k[3] == "compute_center" and len(ev[k]) == 2}
    if not cores:
        continue
    out = []
    for row in (0, 1):
        vals = defaultdict(list)
        for c in cores:
            k = lambda z: (run, c, "TRISC_1", z)
            got = {z: ev[k(z)][row] for z in STAGES if len(ev.get(k(z), [])) == 2}
            for z, (s, e) in got.items():
                vals[z].append(ns(e - s))
            first = min(s for s, _ in got.values())
            vals["add->mean"].append(ns(got["compute_row_mean"][1] - first))
            vals["add->center"].append(ns(got["compute_center"][1] - first))
        out.append(" ".join(f"{z.replace('compute_', '')}={statistics.median(v):.0f}" for z, v in vals.items()))
    print(f"run {run} ({len(cores)} 2-row cores) | row0: {out[0]} | row1: {out[1]}")
