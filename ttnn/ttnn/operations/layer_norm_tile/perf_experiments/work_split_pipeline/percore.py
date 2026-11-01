"""Per-core summary for one run of profile_log_device.csv: kernel span per RISC, block count, read end.

usage: percore.py <csv> <run_index>
"""
import csv
import sys
from collections import defaultdict

path, run_sel = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs, ev = [], defaultdict(lambda: defaultdict(list))
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if run not in runs:
        runs.append(run)
    if run != None:
        ev[run][(int(r[idx["core_x"]]), int(r[idx["core_y"]]))].append(
            (
                r[idx["RISC processor type"]],
                r[idx["zone name"]],
                r[idx["type"]],
                int(r[idx["time[cycles since reset]"]]),
            )
        )
run = runs[run_sel]
cores = ev[run]
starts = [t for c in cores.values() for (risc, z, typ, t) in c if z.endswith("-KERNEL") and typ == "ZONE_START"]
t0 = min(starts)
out = []
for core, evs in cores.items():
    ends = {risc: t for (risc, z, typ, t) in evs if z.endswith("-KERNEL") and typ == "ZONE_END"}
    blocks = sum(1 for (risc, z, typ, t) in evs if z == "compute_row_mean" and typ == "ZONE_START")
    st = min(t for (risc, z, typ, t) in evs if z.endswith("-KERNEL") and typ == "ZONE_START")
    us = lambda t: (t - t0) / freq
    out.append((us(max(ends.values())), core, blocks, us(st), us(ends.get("NCRISC", t0)), us(ends.get("TRISC_1", t0))))
out.sort(reverse=True)
print(f"run {run}: {len(out)} cores, op span {out[0][0]:.1f} us")
for end, core, blocks, st, rd, cp in out[:6]:
    print(f"  {core} blocks {blocks} start {st:.1f} read_end {rd:.1f} compute_end {cp:.1f} end {end:.1f}")
for nb in sorted({o[2] for o in out}):
    sel = [o for o in out if o[2] == nb]
    print(
        f"  blocks={nb}: {len(sel)} cores, max end {max(o[0] for o in sel):.1f}, avg end {sum(o[0] for o in sel)/len(sel):.1f}, avg read_end {sum(o[4] for o in sel)/len(sel):.1f}"
    )
