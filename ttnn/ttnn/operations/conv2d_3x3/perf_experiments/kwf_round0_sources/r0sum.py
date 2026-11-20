# usage: r0sum.py <csv.gz> <calls.txt> : one line per run: call, kernel end max, writer_round0 end p50/max,
# compute's last round-0 chunk wait end p50/max, server/partner kernel starts.
import collections, csv, gzip, sys

path, calls = sys.argv[1], sys.argv[2]
names = [l.split() for l in open(calls)]
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
q = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))]
for n, k in zip(names, big):
    ev = sorted(runs[k], key=lambda e: e[2])
    t0 = min(t for c, r, t, z, ty in ev if z.endswith("-KERNEL") and ty == "ZONE_START")
    us = lambda t: (t - t0) / freq
    st = collections.defaultdict(list)
    sp = collections.defaultdict(list)
    for c, r, t, z, ty in ev:
        key = (c, r, z)
        if ty == "ZONE_START":
            st[key].append(t)
        elif st[key]:
            sp[key].append((us(st[key].pop()), us(t)))
    kend = max(e for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for s, e in v)
    r0 = [v[0][1] for (c, r, z), v in sp.items() if z == "writer_round0"]
    cw = [v[-1][1] for (c, r, z), v in sp.items() if z == "compute_kwf_chunk_wait" and r == "TRISC_0"]
    if not r0 or not cw:
        print(f"{' '.join(n):24s} end {kend:6.2f}  (no kw-fold round-0 zones)")
        continue
    srv = sorted({c for (c, r, z) in sp if z == "server_read"})
    starts = []
    for c in srv:
        s = min(v[0][0] for (cc, r, z), v in sp.items() if cc == c and z.endswith("-KERNEL"))
        fl = sp.get((c, "BRISC", "server_flush"), [(0, 0)])[0][1]
        starts.append(f"{c[0]},{c[1]}:{s:.2f}/{fl:.2f}")
    wst = sorted(
        min(v[0][0] for (cc, r, z), v in sp.items() if cc == c and z.endswith("-KERNEL"))
        for c in {cc for (cc, r, z) in sp if z == "writer_round0"}
    )
    print(
        f"{' '.join(n):24s} ws p50 {q(wst,.5):.2f} max {wst[-1]:.2f}  end {kend:6.2f}  r0 p50 {q(r0,.5):.2f} max {max(r0):.2f}  cw p50 {q(cw,.5):.2f} max {max(cw):.2f}  "
        + " ".join(starts)
    )
