"""Per-run zone summary (p50 / max over cores of per-core summed zone time, us @1.35 GHz) for every run id."""
import csv, sys, collections, statistics as st

rows = list(csv.reader(open(sys.argv[1])))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
for run in sorted(by_run, key=lambda k: int(k) if k.isdigit() else 0):
    open_, dur = {}, collections.defaultdict(list)
    first, last = {}, {}
    for r in by_run[run]:
        core, risc, z, typ, t = (int(r[1]), int(r[2])), r[3].strip(), r[10].strip(), r[11].strip(), int(r[5])
        key = (core, risc, z)
        if typ == "ZONE_START":
            open_.setdefault(key, []).append(t)
            first.setdefault((risc, z), []).append(t)
        elif typ == "ZONE_END" and open_.get(key):
            dur[key].append(t - open_[key].pop())
            last.setdefault((risc, z), []).append(t)
    agg = collections.defaultdict(list)
    for (core, risc, z), ds in dur.items():
        agg[(risc, z)].append(sum(ds))
    t0 = min(int(r[5]) for r in by_run[run])
    print(f"run {run} ({len(by_run[run])} rows)")
    for (risc, z), s in sorted(agg.items()):
        print(
            f"  {risc:8s} {z:22s} cores {len(s):4d} p50 {st.median(s)/1350:8.1f} max {max(s)/1350:8.1f}"
            f"  first-start {(min(first[(risc,z)])-t0)/1350:8.1f} last-end {(max(last.get((risc,z),[t0]))-t0)/1350:8.1f}"
        )
