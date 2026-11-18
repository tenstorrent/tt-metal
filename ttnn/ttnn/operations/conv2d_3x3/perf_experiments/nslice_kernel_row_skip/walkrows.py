# usage: walkrows.py <profile_log_device.csv> <runs/<tag>.jsonl>
# Per dispatch: compute_matmul_walk (TRISC_1) max / p50 over cores, per grid row (virtual y) the row's max walk, and
# the latest TRISC_1 kernel end (us from the dispatch's first kernel start).
import collections, csv, json, statistics, sys

path, log = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(log)]
with open(path) as f:
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
recs = recs[len(recs) - len(big) :]
for rec, k in zip(recs, big):
    ev = sorted(runs[k], key=lambda e: e[2])
    t0 = min(t for c, r, t, z, kd in ev if z.endswith("-KERNEL") and kd == "ZONE_START")
    walk, kend, til = {}, {}, {}
    op = {}
    for core, risc, t, z, kind in ev:
        if risc != "TRISC_1":
            continue
        if kind == "ZONE_START":
            op[(core, z)] = t
        elif (core, z) in op:
            d = (t - op.pop((core, z))) / freq
            if z == "compute_matmul_walk":
                walk[core] = walk.get(core, 0) + d
            elif z == "compute_tilize":
                til[core] = til.get(core, 0) + d
            elif z == "TRISC-KERNEL":
                kend[core] = (t - t0) / freq
    ys = sorted({c[1] for c in walk})
    per_row = " ".join(f"{max(v for c, v in walk.items() if c[1] == y):4.1f}" for y in ys)
    print(
        f"{rec['case']:<18} {rec['variant']:<10} walk max {max(walk.values()):5.2f} p50 {statistics.median(walk.values()):5.2f} "
        f"tilize p50 {statistics.median(til.values()) if til else 0:4.2f}  kend max {max(kend.values()):5.2f} | row max walk: {per_row}"
    )
