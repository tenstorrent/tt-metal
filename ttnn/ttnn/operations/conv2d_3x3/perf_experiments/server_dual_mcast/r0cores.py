# usage: r0cores.py <csv.gz> <run_index> : per core, the end of compute's last round-0 chunk wait (TRISC_0
# compute_kwf_chunk_wait, 3rd occurrence) and the core's TRISC_0 kernel end, grouped by physical row; plus the
# slowest cores.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
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
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda t: (t - t0) / freq
cw = collections.defaultdict(list)
kend = {}
for core, r, t, z, k in ev:
    if r == "TRISC_0" and z == "compute_kwf_chunk_wait" and k == "ZONE_END":
        cw[core].append(us(t))
    if r == "TRISC_0" and z.endswith("-KERNEL") and k == "ZONE_END":
        kend[core] = us(t)
byrow = collections.defaultdict(list)
for core, v in cw.items():
    byrow[core[1]].append((v[-1], kend.get(core, 0)))
for y in sorted(byrow):
    a = sorted(byrow[y])
    print(f"row {y}: n {len(a)} r0 min {a[0][0]:.2f} max {a[-1][0]:.2f}  end max {max(e for _, e in a):.2f}")
worst = sorted(kend.items(), key=lambda kv: -kv[1])[:5]
print("slowest:", [(c, round(e, 2), round(cw[c][-1], 2) if cw.get(c) else None) for c, e in worst])
