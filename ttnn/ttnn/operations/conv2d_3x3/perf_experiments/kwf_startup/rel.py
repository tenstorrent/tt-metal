# usage: rel.py <dev csv[.gz]> <run idx> : per work core, times relative to the core's OWN kernel start (one tile's
# RISCs share a clock; cores may be offset): slab complete (reader_fill end), first round-0 chunk at compute
# (TRISC_0 first compute_kwf_chunk_wait end), Q_1 done (TRISC_0 first side_rounds start, else first weight_wait),
# final end (TRISC_0 compute_kwf_final end), kernel end (TRISC_2). Prints quantiles over cores.
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
first = collections.defaultdict(dict)  # (core, risc, zone, kind) -> first time
last = collections.defaultdict(dict)
for core, risc, t, z, kind in runs[big[idx - 1]]:
    k = (risc, z, kind)
    d = first[core]
    if k not in d or t < d[k]:
        d[k] = t
    d = last[core]
    if k not in d or t > d[k]:
        last[core][k] = t
metrics = collections.defaultdict(list)
for core, d in first.items():
    if ("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END") not in d and (
        "TRISC_0",
        "compute_kwf_matmul_shift",
        "ZONE_START",
    ) not in d:
        continue
    k0 = d.get(("BRISC", "BRISC-KERNEL", "ZONE_START"))
    if k0 is None:
        continue
    rel = lambda t: (t - k0) / freq
    if ("NCRISC", "reader_fill", "ZONE_END") in d:
        metrics["slab"].append(rel(d[("NCRISC", "reader_fill", "ZONE_END")]))
    if ("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END") in d:
        metrics["w0"].append(rel(d[("TRISC_0", "compute_kwf_chunk_wait", "ZONE_START")]))
    if ("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END") in d:
        metrics["mac0"].append(rel(d[("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END")]))
    if ("BRISC", "writer_weight_fetch", "ZONE_START") in d:
        metrics["wfetch"].append(rel(d[("BRISC", "writer_weight_fetch", "ZONE_START")]))
    if ("BRISC", "writer_fill_early", "ZONE_END") in d:
        metrics["wearly"].append(rel(d[("BRISC", "writer_fill_early", "ZONE_END")]))
    if ("TRISC_0", "compute_kwf_matmul_shift", "ZONE_START") in d:
        metrics["tstart"].append(rel(d[("TRISC_0", "compute_kwf_matmul_shift", "ZONE_START")]))
    l = last[core]
    if ("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END") in l:
        metrics["w0_all"].append(rel(l[("TRISC_0", "compute_kwf_chunk_wait", "ZONE_END")]))
    if ("TRISC_0", "compute_kwf_final", "ZONE_START") in d:
        metrics["final0"].append(rel(d[("TRISC_0", "compute_kwf_final", "ZONE_START")]))
    if ("TRISC_2", "TRISC-KERNEL", "ZONE_END") in l:
        metrics["end"].append(rel(l[("TRISC_2", "TRISC-KERNEL", "ZONE_END")]))
for m in ("wearly", "wfetch", "tstart", "slab", "w0", "mac0", "w0_all", "final0", "end"):
    v = sorted(metrics[m])
    if not v:
        continue
    n = len(v)
    print(
        f"  {m:7s} n={n:3d} " + " ".join(f"q{int(q*100):<3d}{v[min(n-1,int(q*n))]:.2f}" for q in (0, 0.1, 0.5, 0.9, 1))
    )
