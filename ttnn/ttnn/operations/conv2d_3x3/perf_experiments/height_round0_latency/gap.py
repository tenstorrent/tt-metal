# usage: gap.py <csv> <run_index>: per work core, writer kernel start, first weight_reserve start, first weight_fetch
# start, first fill start / end, reader kernel start, fill_reserve end, first fwd_wait start/end, fill end.
import collections, csv, sys, statistics as st

path, idx = sys.argv[1], int(sys.argv[2])
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
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda c: (c - t0) / freq
z = collections.defaultdict(list)
stack = collections.defaultdict(list)
for core, risc, t, name, kind in ev:
    if kind == "ZONE_START":
        stack[(core, risc, name)].append(t)
    elif stack[(core, risc, name)]:
        s = stack[(core, risc, name)].pop()
        z[(core, risc, name)].append((us(s), us(t)))
cores = sorted({k[0] for k in z if k[2] == "writer_fill"})
names = ["wK", "wRes", "wFetch", "fill0s", "fill0e", "fill1s", "fill2s", "rK", "rResE", "fwd0s", "fwd0e", "rFillE"]
data = collections.defaultdict(list)
for c in cores:
    g = lambda r, n: sorted(z.get((c, r, n), []))
    fl = g("BRISC", "writer_fill")
    fw = g("NCRISC", "reader_fill_fwd_wait")
    v = [
        g("BRISC", "BRISC-KERNEL")[0][0],
        g("BRISC", "writer_weight_reserve")[0][0],
        g("BRISC", "writer_weight_fetch")[0][0],
        fl[0][0],
        fl[0][1],
        fl[1][0] if len(fl) > 1 else -1,
        fl[2][0] if len(fl) > 2 else -1,
        g("NCRISC", "NCRISC-KERNEL")[0][0],
        g("NCRISC", "reader_fill_reserve")[0][1],
        fw[0][0] if fw else -1,
        fw[0][1] if fw else -1,
        g("NCRISC", "reader_fill")[0][1],
    ]
    for n, x in zip(names, v):
        data[n].append(x)
print(f"{len(cores)} cores with writer fill")
for n in names:
    v = [x for x in data[n] if x >= 0]
    print(f"  {n:7s} p50 {st.median(v):5.2f} min {min(v):5.2f} max {max(v):5.2f}")
