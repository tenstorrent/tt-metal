"""Per-core tail after the last input: usage tailz.py <profile_log_device.csv>...
For every core: A_last = end of wr_a_receive / wr_a_inject (BRISC), B_last = end of the last rd_barrier (NCRISC),
in_last = max of both; T1 = TRISC_1 kernel end; W = BRISC kernel end. Prints percentiles of in_last, T1-in_last, W-in_last
and the op span (max W - min start)."""
import csv, sys, collections, statistics

for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))[2:]
    by_run = collections.defaultdict(list)
    for r in rows:
        by_run[r[7].strip()].append(r)
    run = max(by_run, key=lambda k: len(by_run[k]))
    R = by_run[run]
    t0 = min(int(r[5]) for r in R)
    ends = collections.defaultdict(dict)
    for r in R:
        core, risc, z, typ, t = (
            (int(r[1]), int(r[2])),
            r[3].strip(),
            r[10].strip(),
            r[11].strip(),
            (int(r[5]) - t0) / 1350,
        )
        if typ == "ZONE_END":
            k = (risc, z)
            ends[core][k] = max(ends[core].get(k, 0), t)
    il, t1, w = [], [], []
    for c, e in ends.items():
        a = max(e.get(("BRISC", "wr_a_receive"), 0), e.get(("BRISC", "wr_a_inject"), 0))
        b = e.get(("NCRISC", "rd_barrier"), 0)
        x = max(a, b)
        il.append(x)
        t1.append(e[("TRISC_1", "TRISC-KERNEL")] - x)
        w.append(e[("BRISC", "BRISC-KERNEL")] - x)
    q = lambda v: f"{min(v):6.1f}/{statistics.median(v):6.1f}/{max(v):6.1f}"
    print(
        f"{path.split('/')[-1]:28s} in_last {q(il)}  T1-in {q(t1)}  W-in {q(w)}  span {max(e[('BRISC','BRISC-KERNEL')] for e in ends.values()):.1f}"
    )
