# usage: walks.py <profile_log_device.csv[.gz]> <run_index> : per core compute_matmul_walk start / dur, TRISC end (us),
# sorted by TRISC end; summary percentiles.
import collections, csv, gzip, statistics, sys

path, idx = sys.argv[1], int(sys.argv[2])
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for _, _, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
ks, ke, ws, we = {}, collections.defaultdict(int), {}, {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL"):
        if kind == "ZONE_START":
            ks[core] = min(ks.get(core, t), t)
        else:
            ke[core] = max(ke[core], t)
    if z == "compute_matmul_walk" and risc == "TRISC_0":
        if kind == "ZONE_START":
            ws[core] = t
        else:
            we[core] = t
cores = sorted(ke, key=lambda c: -ke[c])
print(f"run {big[idx-1]}: {len(cores)} cores; kernel end max {us(ke[cores[0]]):.2f}")
for c in cores if len(sys.argv) > 3 else cores[:12]:
    print(
        f"  {c}: kstart {us(ks[c]):.2f} walk {us(ws.get(c,t0)):.2f}..{us(we.get(c,t0)):.2f} ({(we.get(c,0)-ws.get(c,0))/freq:.2f}) end {us(ke[c]):.2f}"
    )
d = [(we[c] - ws[c]) / freq for c in ws if c in we]
s = [us(ws[c]) for c in ws]
print(
    f"  walk dur p50 {statistics.median(d):.2f} max {max(d):.2f} min {min(d):.2f}; walk start p50 {statistics.median(s):.2f} max {max(s):.2f}"
)
