# usage: zsum.py <dev csv(.gz)> <calls file>: per run (in call order), the kw-fold compute zones (TRISC_1 p50 / max, us)
# and the TRISC kernel span, joined with the call log.
import collections, csv, gzip, statistics, sys

RISC = __import__("os").environ.get("ZRISC", "TRISC_1")
path, calls = sys.argv[1], [l.split() for l in open(sys.argv[2])]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
ZONES = [
    "compute_kwf_matmul_shift",
    "compute_kwf_side_rounds",
    "compute_kwf_fused_rows",
    "compute_kwf_final",
    "compute_kwf_halo_wait",
    "compute_kwf_weight_wait",
]
for k, c in zip(big, calls):
    st, sums = collections.defaultdict(list), collections.defaultdict(float)
    kst, ken = {}, {}
    for core, risc, t, z, kind in sorted(runs[k], key=lambda e: e[2]):
        if z.endswith("-KERNEL"):
            if kind == "ZONE_START":
                kst.setdefault(core, t)
            else:
                ken[core] = max(ken.get(core, 0), t)
        if risc != RISC:
            continue
        key = (core, z)
        if kind == "ZONE_START":
            st[key].append(t)
        elif kind == "ZONE_END" and st[key]:
            sums[key] += t - st[key].pop()
    out = []
    for zname in ZONES:
        v = [cyc / freq for (core, z), cyc in sums.items() if z == zname]
        if v:
            out.append(f"{zname.replace('compute_kwf_', '')} {statistics.median(v):.2f}/{max(v):.2f}")
    span = (max(ken.values()) - min(kst.values())) / freq if kst else 0
    print(" ".join(c), f"span {span:.2f}", " | ".join(out))
