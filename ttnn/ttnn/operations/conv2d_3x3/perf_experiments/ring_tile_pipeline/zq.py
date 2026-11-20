# usage: zq.py <device csv[.gz]> <run_index> <x,y> <risc> <zone substring>: start/end (us from the run's earliest kernel start)
import collections, csv, gzip, sys

path, idx, core, risc, zs = (
    sys.argv[1],
    int(sys.argv[2]),
    tuple(int(v) for v in sys.argv[3].split(",")),
    sys.argv[4],
    sys.argv[5],
)
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    r = csv.reader(f)
    cols = [c.strip() for c in next(r)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for row in r:
        if len(row) < len(cols):
            continue
        runs[int(row[ix["run host ID"]])].append(
            (
                (int(row[ix["core_x"]]), int(row[ix["core_y"]])),
                row[ix["RISC processor type"]].strip(),
                int(row[ix["time[cycles since reset]"]]),
                row[ix["zone name"]].strip(),
                row[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for _, _, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
st = {}
out = []
for c, rr, t, z, k in ev:
    if c != core or rr != risc or zs not in z:
        continue
    if k == "ZONE_START":
        st[z] = t
    elif z in st:
        out.append(f"{z}[{(st.pop(z)-t0)/freq:.2f}-{(t-t0)/freq:.2f}]")
print(" ".join(out))
