"""Per run: per compute-row-count class, mean TRISC_1 zone durations (us per core) and phase ends (us from op start)
of the latest-ending core. usage: rowzones.py <device csv> <run index,...>"""
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
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    k = (run, core, r[idx["RISC processor type"]], r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        ev[run][core].append((k[2], k[3], open_.pop(k), t))
for s in sel:
    run = runs[s]
    cores = ev[run]
    t0 = min(st for c in cores for (_, z, st, _) in cores[c] if z.endswith("KERNEL"))
    end = lambda c: max(e[3] for e in cores[c] if e[1].endswith("KERNEL"))
    nrows = lambda c: sum(1 for e in cores[c] if e[0] == "TRISC_1" and e[1] == "compute_add_and_stats")
    cls = defaultdict(list)
    for c in cores:
        cls[nrows(c)].append(c)
    print(f"run#{s} ({run}) end {(max(end(c) for c in cores)-t0)/freq:.2f} us")
    for n, cs in sorted(cls.items()):
        zs = defaultdict(list)
        for c in cs:
            acc = defaultdict(float)
            for risc, z, st, e in cores[c]:
                if risc == "TRISC_1" and z.startswith("compute_"):
                    acc[z] += (e - st) / freq
            for z, v in acc.items():
                zs[z].append(v)
        ends = sorted((end(c) - t0) / freq for c in cs)
        print(
            f"  rows={n} cores={len(cs)} end max {ends[-1]:.2f} p50 {ends[len(ends)//2]:.2f} | "
            + " ".join(f"{z[8:]}={statistics.mean(v):.2f}" for z, v in sorted(zs.items()))
        )
    c = max(cores, key=end)
    tl = defaultdict(lambda: [1e18, 0])
    for risc, z, st, e in cores[c]:
        if z.endswith("KERNEL") or risc in ("TRISC_0", "TRISC_2"):
            continue
        tl[(risc, z)][0] = min(tl[(risc, z)][0], st)
        tl[(risc, z)][1] = max(tl[(risc, z)][1], e)
    print(
        f"  latest core {c} rows={nrows(c)}: "
        + "; ".join(
            f"{z} {(a-t0)/freq:.1f}-{(b-t0)/freq:.1f}" for (r, z), (a, b) in sorted(tl.items(), key=lambda kv: kv[1][0])
        )
    )
