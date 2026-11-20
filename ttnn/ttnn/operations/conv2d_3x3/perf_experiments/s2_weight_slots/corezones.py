# usage: corezones.py <csv[.gz]> <run_index> x,y : every zone of one core, per RISC, in start order (us after run t0)
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(","))
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
stack = collections.defaultdict(list)
out = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if core != want:
        continue
    if kind == "ZONE_START":
        stack[(risc, z)].append(t)
    elif kind == "ZONE_END" and stack[(risc, z)]:
        s = stack[(risc, z)].pop()
        out[risc].append((us(s), us(t), z))
for risc in sorted(out):
    print(risc, "  ".join("%s[%.2f-%.2f]" % (z, s, e) for s, e, z in sorted(out[risc]) if not z.endswith("-FW")))
