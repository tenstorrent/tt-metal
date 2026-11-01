"""Mean TRISC_1 zone durations of the LAST compute row of the heavy cores (resident: data landed), per run.
usage: row1.py <device csv> <run index,...>"""
import csv, sys, statistics
from collections import defaultdict

path, sel = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs, open_, ev = [], {}, defaultdict(lambda: defaultdict(list))
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if run not in runs:
        runs.append(run)
    if r[idx["RISC processor type"]] != "TRISC_1":
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    k = (run, core, r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        ev[run][core].append((k[2], (t - open_.pop(k)) / freq))
for s in sel:
    cores = ev[runs[s]]
    n = max(sum(1 for z, _ in v if z == "compute_add_and_stats") for v in cores.values())
    acc = defaultdict(list)
    for v in cores.values():
        if sum(1 for z, _ in v if z == "compute_add_and_stats") != n:
            continue
        seen = defaultdict(int)
        for z, d in v:
            seen[z] += 1
            if seen[z] == n and z.startswith("compute_"):
                acc[z].append(d)
    tot = sum(statistics.median(v) for z, v in acc.items() if z != "compute_wait_affine")
    print(
        f"run#{s} rows={n}: "
        + " ".join(f"{z[8:]}={statistics.median(v):.2f}" for z, v in acc.items())
        + f" | sum {tot:.2f}"
    )
