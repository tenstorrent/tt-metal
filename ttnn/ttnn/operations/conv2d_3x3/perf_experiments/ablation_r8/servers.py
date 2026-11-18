import collections, csv, sys

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
srv = sorted({e[0] for e in ev if e[3].startswith("server_")})
for core in srv:
    st = collections.defaultdict(list)
    out = []
    for c, risc, t, z, kind in ev:
        if c != core:
            continue
        if kind == "ZONE_START":
            st[(risc, z)].append(t)
        elif st[(risc, z)]:
            s = st[(risc, z)].pop()
            out.append((risc, us(s), us(t), z))
    for risc in ("NCRISC", "BRISC"):
        items = sorted([o for o in out if o[0] == risc], key=lambda o: o[1])
        print(
            core,
            risc,
            " ".join(f"{z.replace('server_','s.')}[{a:.1f}-{b:.1f}]" for _, a, b, z in items if b - a >= 0.1),
        )
