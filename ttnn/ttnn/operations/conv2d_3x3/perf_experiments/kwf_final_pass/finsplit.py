# usage: finsplit.py <profile_log_device.csv[.gz]> [run_index ...]: for each run (1-based among runs with >= 8 cores),
# the critical core's (last TRISC end) kw-fold final-pass zones per TRISC, in us relative to the earliest kernel start,
# plus the per-zone p50 / max duration over all cores.
import collections, csv, gzip, statistics, sys

path = sys.argv[1]
want_idx = [int(a) for a in sys.argv[2:]]
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
KEEP = (
    "kfz_",
    "compute_kwf_final",
    "compute_kwf_halo_wait",
    "compute_kwf_matmul_shift",
    "compute_kwf_side_rounds",
    "TRISC-KERNEL",
)
for idx in want_idx or range(1, len(big) + 1):
    ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
    t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
    us = lambda c: (c - t0) / freq
    stack = collections.defaultdict(list)
    spans = collections.defaultdict(list)  # (core, risc) -> [(zone, s, e)]
    for core, risc, t, z, kind in ev:
        if kind == "ZONE_START":
            stack[(core, risc, z)].append(t)
        elif stack[(core, risc, z)]:
            s = stack[(core, risc, z)].pop()
            spans[(core, risc)].append((z, us(s), us(t)))
    kend = {}
    for (core, risc), sp in spans.items():
        for z, s, e in sp:
            if z == "TRISC-KERNEL":
                kend[core] = max(kend.get(core, 0), e)
    crit = max(kend, key=kend.get)
    print(f"=== run {idx}: critical core {crit} TRISC end {kend[crit]:.2f} us")
    for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
        sp = sorted([x for x in spans[(crit, risc)] if x[0].startswith(KEEP)], key=lambda x: x[1])
        print(
            f"  {risc}: "
            + "  ".join(
                f"{z.replace('compute_kwf_','c.').replace('TRISC-KERNEL','K')}[{s:.2f}-{e:.2f}]"
                for z, s, e in sp
                if z not in ("compute_kwf_matmul_shift", "compute_kwf_side_rounds")
            )
        )
    # p50/max per zone per risc across cores (sum over executions per core)
    agg = collections.defaultdict(list)
    for (core, risc), sp in spans.items():
        tot = collections.defaultdict(float)
        for z, s, e in sp:
            if z.startswith("kfz_") or z == "compute_kwf_final":
                tot[z] += e - s
        for z, v in tot.items():
            agg[(risc, z)].append(v)
    line = []
    for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
        line.append(
            risc
            + ": "
            + " ".join(
                f"{z[4:] if z.startswith('kfz_') else 'final'}={statistics.median(v):.2f}/{max(v):.2f}"
                for (r, z), v in sorted(agg.items())
                if r == risc
            )
        )
    print("   p50/max  " + "\n   p50/max  ".join(line))
