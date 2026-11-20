# usage: kstart.py <csv.gz> <run_index> : BRISC kernel start (us from the run's earliest start) per core, as a grid.
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
        if r[ix["zone name"]].strip().endswith("-KERNEL") and r[ix["type"]].strip() == "ZONE_START":
            runs[int(r[ix["run host ID"]])].append(
                (
                    (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                    r[ix["RISC processor type"]].strip(),
                    int(r[ix["time[cycles since reset]"]]),
                )
            )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = runs[big[idx - 1]]
t0 = min(t for _, _, t in ev)
st = {}
for c, r, t in ev:
    st[c] = min(st.get(c, t), t)
xs = sorted({c[0] for c in st})
ys = sorted({c[1] for c in st})
print("      " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:4d}  " + " ".join(f"{(st[(x,y)]-t0)/freq:5.2f}" if (x, y) in st else "    -" for x in xs))
