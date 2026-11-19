# usage: zsum.py <dev csv[.gz]> <run idx> [risc]: per zone, median over cores of (count, total us, first start rel to
# own kernel start), for one RISC (default TRISC_0).
import collections, csv, gzip, statistics, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = sys.argv[3] if len(sys.argv) > 3 else "TRISC_0"
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
k0 = {}
for core, risc, t, z, kind in ev:
    if risc == "BRISC" and z == "BRISC-KERNEL" and kind == "ZONE_START":
        k0.setdefault(core, t)
stack = collections.defaultdict(list)
tot = collections.defaultdict(lambda: collections.defaultdict(float))
cnt = collections.defaultdict(lambda: collections.defaultdict(int))
for core, risc, t, z, kind in ev:
    if risc != want or core not in k0:
        continue
    if kind == "ZONE_START":
        stack[(core, z)].append(t)
    elif stack[(core, z)]:
        s = stack[(core, z)].pop()
        tot[z][core] += (t - s) / freq
        cnt[z][core] += 1
for z in sorted(tot):
    v = sorted(tot[z].values())
    c = sorted(cnt[z].values())
    print(f"  {z:28s} n={c[len(c)//2]:3d} total med {v[len(v)//2]:.2f} q90 {v[int(.9*len(v))]:.2f} max {v[-1]:.2f}")
