# usage: r0tl.py <profile_log_device.csv[.gz]> <run_index (1-based among runs with >= 8 cores)> [--all]
# Round-0 timeline of a kw-fold server-fed run (us from the run's earliest kernel start):
#  - per server core (cores with a server_read zone): kernel starts, per-round read spans (NCRISC server_read) and
#    multicast chains (BRISC server_send), flushes;
#  - receivers: writer_round0 end (round 0 complete and pushed), compute's last round-0 chunk wait end
#    (compute_kwf_chunk_wait, 3rd occurrence), first-chunk wait end, kernel end; quantiles over cores; slowest cores.
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
spans = collections.defaultdict(list)  # (core, risc, zone) -> [(s, e)]
st = collections.defaultdict(list)
for core, risc, t, z, k in ev:
    key = (core, risc, z)
    if k == "ZONE_START":
        st[key].append(t)
    elif k == "ZONE_END" and st[key]:
        spans[key].append((us(st[key].pop()), us(t)))
servers = sorted({c for (c, r, z) in spans if z == "server_read"})
kend = {}
kstart = {}
for (c, r, z), v in spans.items():
    if z.endswith("-KERNEL"):
        kend[c] = max(kend.get(c, 0), max(e for _, e in v))
        kstart[(c, r)] = min(s for s, _ in v)
end_all = max(kend.values())
print(f"run {big[idx-1]}: kernel end max {end_all:.2f} us")
fmt = lambda v: " ".join(f"[{a:.2f}-{b:.2f}]" for a, b in v)
for c in servers:
    ks = {r: kstart.get((c, r)) for r in ("BRISC", "NCRISC")}
    print(f"server {c}: start B {ks['BRISC']:.2f} N {ks['NCRISC']:.2f}  end {kend.get(c, 0):.2f}")
    print(f"   read  {fmt(spans[(c, 'NCRISC', 'server_read')][:3])}")
    print(f"   wait_read {fmt(spans[(c, 'BRISC', 'server_wait_read')][:6])}")
    print(f"   send  {fmt(spans[(c, 'BRISC', 'server_send')][:6])}")
    print(
        f"   bias  {fmt(spans[(c, 'BRISC', 'server_send_bias')][:1])}  flush {fmt(spans[(c, 'BRISC', 'server_flush')][:3])}"
    )
q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]


def quant(name, vals):
    vals = sorted(vals)
    if not vals:
        return
    print(
        f"{name:28s} n {len(vals):3d}  p10 {q(vals,.1):.2f} p50 {q(vals,.5):.2f} p90 {q(vals,.9):.2f} max {vals[-1]:.2f}"
    )


r0 = {c: v[0][1] for (c, r, z), v in spans.items() if z == "writer_round0" and r == "BRISC"}
cw = {c: v for (c, r, z), v in spans.items() if z == "compute_kwf_chunk_wait" and r == "TRISC_0"}
quant("writer_round0 end", r0.values())
quant("chunk wait 0 end", [v[0][1] for v in cw.values() if len(v) > 0])
quant("chunk wait last end", [v[-1][1] for v in cw.values()])
quant("chunk wait total", [sum(e - s for s, e in v) for v in cw.values()])
fill = {c: v for (c, r, z), v in spans.items() if z == "reader_fill" and r == "NCRISC"}
quant("reader_fill end", [v[-1][1] for v in fill.values()])
quant("kernel end (receivers)", [kend[c] for c in r0])
if "--all" in sys.argv:
    byrow = collections.defaultdict(list)
    for c, v in cw.items():
        byrow[c[1]].append((v[-1][1], c[0]))
    for y in sorted(byrow):
        a = sorted(byrow[y])
        print(f"  row {y}: n {len(a)} r0 last chunk min {a[0][0]:.2f} max {a[-1][0]:.2f} (x {a[-1][1]})")
worst = sorted(((kend[c], c) for c in r0), reverse=True)[:4]
print("slowest:", [(c, round(e, 2), round(cw[c][-1][1], 2) if c in cw else None) for e, c in worst])
