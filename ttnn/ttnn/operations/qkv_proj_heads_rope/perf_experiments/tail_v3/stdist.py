"""st_issue per call index: median/max start and duration (us) per RISC, split by number of calls (core type)."""
import collections, csv, statistics, sys

path = sys.argv[1]
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in list(csv.reader(f))[1:] if len(r) >= 12]
run = [r[7].strip() for r in rows if r[10].strip() == "compute_last_k_epilogue"][-1]
rows = [r for r in rows if r[7].strip() == run]
seq = collections.defaultdict(list)
t0 = None
for r in rows:
    core = (int(r[1]), int(r[2]))
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    if zone == "st_issue":
        if typ == "ZONE_START":
            seq[(core, risc)].append([t, None])
        else:
            seq[(core, risc)][-1][1] = t
groups = collections.defaultdict(lambda: collections.defaultdict(list))
ncalls = {core: sum(len(c) for (cc, _), c in seq.items() if cc == core) for core, _ in seq}
for (core, risc), calls in seq.items():
    for i, (a, b) in enumerate(calls):
        groups[(risc, ncalls[core])][i].append(((a - t0) / freq, (b - a) / freq, (b - t0) / freq))
for (risc, n), d in sorted(groups.items()):
    print(risc, "units/core", n, "cores", len(d[0]))
    for i, v in sorted(d.items()):
        st = [a for a, _, _ in v]
        du = [b for _, b, _ in v]
        en = [c for _, _, c in v]
        print(
            f"   #{i} start med {statistics.median(st):6.2f}  dur med {statistics.median(du):5.2f} max {max(du):5.2f} end med {statistics.median(en):6.2f} max {max(en):6.2f}"
        )
