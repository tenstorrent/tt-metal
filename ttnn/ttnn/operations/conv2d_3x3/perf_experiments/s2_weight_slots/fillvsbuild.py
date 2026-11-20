# usage: fillvsbuild.py <csv[.gz]> <run_index>: per compute core, fill end (reader_fill / writer_fill ends), tap-build
# end per TRISC (first compute_tilize zone), walk start / end, kernel end; sorted by walk start; plus correlations.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda c: (c - t0) / freq
z = collections.defaultdict(list)
st = collections.defaultdict(list)
for core, risc, t, name, kind in ev:
    if kind == "ZONE_START":
        st[(core, risc, name)].append(t)
    elif kind == "ZONE_END" and st[(core, risc, name)]:
        z[(core, risc, name)].append((us(st[(core, risc, name)].pop()), us(t)))
out = []
for core in sorted({k[0] for k in z}):
    w = z.get((core, "TRISC_0", "compute_matmul_walk"))
    if not w:
        continue
    fill = max(
        [e for _, e in z.get((core, "NCRISC", "reader_fill"), [])]
        + [e for _, e in z.get((core, "BRISC", "writer_fill"), [])]
        + [0]
    )
    tb = [min(z[(core, r, "compute_tilize")])[1] for r in ("TRISC_0", "TRISC_1", "TRISC_2")]
    end = max(e for _, e in z[(core, "TRISC_2", "TRISC-KERNEL")])
    out.append((tb[1], core, fill, tb, w[0], end))
out.sort()
print("core          fill_end  build_end(T0/T1/T2)   build-fill(T1)  walk          end")
for b1, core, fill, tb, w, end in out[:: max(1, len(out) // 20)] + [out[-1]]:
    print(
        f"{str(core):12s} {fill:6.2f}   {tb[0]:5.2f}/{tb[1]:5.2f}/{tb[2]:5.2f}   {b1 - fill:6.2f}        {w[0]:5.2f}-{w[1]:5.2f}  {end:5.2f}"
    )
import statistics

fills = [o[2] for o in out]
b1s = [o[0] for o in out]
print("fill_end min/p50/max %.2f %.2f %.2f" % (min(fills), statistics.median(fills), max(fills)))
print("build_end(T1) min/p50/max %.2f %.2f %.2f" % (min(b1s), statistics.median(b1s), max(b1s)))
print("corr(fill_end, build_end) %.2f" % statistics.correlation(fills, b1s))
