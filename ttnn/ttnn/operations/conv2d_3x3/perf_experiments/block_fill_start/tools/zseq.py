# usage: zseq.py <csv> <run idx> <zone> <x,y> [risc]: every occurrence of a zone on one core (start-end, us)
import collections, csv, sys

path, idx, zone, core = sys.argv[1], int(sys.argv[2]), sys.argv[3], tuple(int(v) for v in sys.argv[4].split(","))
want_risc = sys.argv[5] if len(sys.argv) > 5 else None
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
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
st = []
out = []
for c, r, t, z, k in ev:
    if c != core or z != zone or (want_risc and r != want_risc):
        continue
    if k == "ZONE_START":
        st.append(t)
    elif st:
        s = st.pop()
        out.append(f"{(s-t0)/freq:.2f}-{(t-t0)/freq:.2f}")
print(core, zone, " ".join(out))
