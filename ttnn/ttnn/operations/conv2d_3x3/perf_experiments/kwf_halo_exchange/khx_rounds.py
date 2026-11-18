# usage: khx_rounds.py <profile_log_device.csv> <run_index>
# Per-core kw-fold round phases on TRISC_0 (us): round s compute = next weight wait start (or matmul_shift end) - this
# round's weight wait end; prints the medians and maxima over cores, and the final pass.
import collections, csv, statistics, sys

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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL"))
us = lambda c: (c - t0) / freq
per = collections.defaultdict(lambda: collections.defaultdict(list))
for core, risc, t, z, kind in ev:
    if risc == "TRISC_0" and z in ("compute_kwf_weight_wait", "compute_kwf_matmul_shift", "compute_kwf_final"):
        per[core][(z, kind)].append(us(t))
stats = collections.defaultdict(list)
for core, d in per.items():
    ww_s, ww_e = d[("compute_kwf_weight_wait", "ZONE_START")], d[("compute_kwf_weight_wait", "ZONE_END")]
    ms_e = d[("compute_kwf_matmul_shift", "ZONE_END")]
    fs, fe = d[("compute_kwf_final", "ZONE_START")], d[("compute_kwf_final", "ZONE_END")]
    if len(ww_s) != 3 or not ms_e or not fe:
        continue
    ends = [ww_s[1], ww_s[2], ms_e[0]]
    for s in range(3):
        stats[f"round{s}_land"].append(ww_e[s])
        stats[f"round{s}_compute"].append(ends[s] - ww_e[s])
    stats["final"].append(fe[0] - fs[0])
    stats["end"].append(fe[0])
for k, v in stats.items():
    print(f"  {k:16s} median {statistics.median(v):6.2f}  max {max(v):6.2f}  min {min(v):6.2f}")
