# usage: starts.py <csv> <run>: per core, NCRISC / BRISC / TRISC_0 kernel start, reader_fill start, writer_fill_early
# start, first compute_ts_kh start (us after the run's earliest kernel start), for grid rows 4..9.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rd:
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
ev = runs[big[idx - 1]]
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
first = {}
for c, r, t, z, k in ev:
    if k == "ZONE_START":
        key = (c, r, z)
        first[key] = min(first.get(key, t), t)
us = lambda v: (v - t0) / freq if v is not None else float("nan")
print("core      NC-K   BR-K   T0-K  r.fill w.early ts_kh0")
for c in sorted({e[0] for e in ev}):
    if c[1] not in (4, 5, 6, 7, 8, 9):
        continue
    g = lambda r, z: us(first.get((c, r, z)))
    print(
        f"{str(c):9s} {g('NCRISC','NCRISC-KERNEL'):5.2f}  {g('BRISC','BRISC-KERNEL'):5.2f}  {g('TRISC_0','TRISC-KERNEL'):5.2f}  {g('NCRISC','reader_fill'):5.2f}  {g('BRISC','writer_fill_early'):5.2f}  {g('TRISC_0','compute_ts_kh'):5.2f}"
    )
