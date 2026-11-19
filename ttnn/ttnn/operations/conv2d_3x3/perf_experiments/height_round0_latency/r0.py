# usage: r0.py <csv> <run_index>: per work core, the round-0 chain: kernel start, writer round-0 flag seen
# (end of the first w.weight_fetch), first writer fill issue, reader fill end, compute round-0 start (end of the first
# c.kwf_weight_wait on TRISC_0), compute round-0 end (start of the second), compute end. Sorted by round-0 start.
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
cores = sorted({k[0] for k in z if k[2] == "compute_kwf_weight_wait"})
rows = []
for c in cores:
    g = lambda r, n: sorted(z.get((c, r, n), []))
    ks = g("BRISC", "BRISC-KERNEL")
    wf = g("BRISC", "writer_weight_fetch")
    fl = g("BRISC", "writer_fill")
    rf = g("NCRISC", "reader_fill")
    ww = g("TRISC_0", "compute_kwf_weight_wait")
    ck = g("TRISC_2", "TRISC-KERNEL")
    rows.append(
        (
            c,
            ks[0][0] if ks else -1,
            wf[0][1] if wf else -1,
            fl[0][0] if fl else -1,
            rf[0][1] if rf else -1,
            ww[0][1],
            ww[1][0] if len(ww) > 1 else -1,
            ck[0][1] if ck else -1,
        )
    )
rows.sort(key=lambda r: r[5])
print(f"run {big[idx-1]}: {len(rows)} cores. cols: core kstart w.r0flag w.fill0 r.fill_end c.r0start c.r0end c.end")
for r in rows[:3] + [None] + rows[len(rows) // 2 - 1 : len(rows) // 2 + 2] + [None] + rows[-4:]:
    print("  ..." if r is None else "  %-9s" % (r[0],) + " ".join("%6.2f" % v for v in r[1:]))
for i, n in enumerate(["kstart", "w.r0flag", "w.fill0", "r.fill_end", "c.r0start", "c.r0end", "c.end"]):
    v = [r[i + 1] for r in rows if r[i + 1] >= 0]
    if v:
        print(f"  {n:10s} p50 {st.median(v):5.2f} max {max(v):5.2f} min {min(v):5.2f}")
