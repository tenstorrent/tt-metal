"""zone_stages.py <profile_log_device.csv> ...: per-launch stage cycles of a one-core g6_clone zone build.

For each RISC: kernel start -> first g6c zone start (prologue), each g6c zone's duration, last g6c zone end ->
kernel end (epilogue). Medians over the launches after the first. Zone overhead (~45-50 cycles per zone) is in
every number; compare variants, not absolutes."""
import csv
import statistics
import sys
from collections import defaultdict


def stages(path):
    rows = list(csv.reader(open(path)))[2:]
    per = defaultdict(list)
    for r in rows:
        per[(r[7], r[1], r[2], r[3])].append((int(r[5]), r[10].strip(), r[11].strip()))
    out = defaultdict(lambda: defaultdict(list))
    launches = sorted({k[0] for k in per}, key=int)[1:]
    for (run, x, y, risc), ev in per.items():
        if run not in launches:
            continue
        ev.sort()
        kstart = next(t for t, z, ty in ev if z.endswith("-KERNEL") and ty == "ZONE_START")
        kend = next(t for t, z, ty in ev if z.endswith("-KERNEL") and ty == "ZONE_END")
        zs = [(t, z, ty) for t, z, ty in ev if z.startswith("g6c_")]
        if not zs:
            continue
        out[risc]["prologue"].append(zs[0][0] - kstart)
        opened = {}
        for t, z, ty in zs:
            if ty == "ZONE_START":
                opened[z] = t
            else:
                out[risc][z].append(t - opened.pop(z))
        out[risc]["epilogue"].append(kend - zs[-1][0])
        out[risc]["kernel"].append(kend - kstart)
    return out


for path in sys.argv[1:]:
    print(path.split("/")[-1])
    for risc, st in sorted(stages(path).items()):
        print("  " + risc + ": " + ", ".join(f"{k} {statistics.median(v):.0f}" for k, v in st.items()))
