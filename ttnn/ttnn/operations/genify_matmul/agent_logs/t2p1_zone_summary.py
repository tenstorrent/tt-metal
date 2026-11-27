"""Tournament-2 Perf 1: per-zone summary over cores (us from the op's earliest start): first-start / last-end
percentiles and per-core total, per RISC. usage: t2p1_zone_summary.py <profile_log_device.csv>"""
import csv, sys, collections, statistics

rows = list(csv.reader(open(sys.argv[1])))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
R = by_run[run]
t0 = min(int(r[5]) for r in R)
open_ = {}
first, last, tot, cnt = {}, {}, collections.defaultdict(int), collections.Counter()
for r in R:
    core, risc, z, typ, t = (int(r[1]), int(r[2])), r[3].strip(), r[10].strip(), r[11].strip(), int(r[5]) - t0
    key = (core, risc, z)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(t)
        first.setdefault(key, t)
    elif typ == "ZONE_END" and open_.get(key):
        s = open_[key].pop()
        tot[key] += t - s
        cnt[key] += 1
        last[key] = t
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for key in tot:
    core, risc, z = key
    agg[(risc, z)]["first"].append(first[key] / 1350)
    agg[(risc, z)]["last"].append(last[key] / 1350)
    agg[(risc, z)]["tot"].append(tot[key] / 1350)
    agg[(risc, z)]["n"].append(cnt[key])
print(f"{'risc':8s} {'zone':22s} cores  first(min/med/max)      last(min/med/max)       total(min/med/max)    n")
for (risc, z), d in sorted(agg.items()):
    f = lambda v: f"{min(v):6.1f}/{statistics.median(v):6.1f}/{max(v):6.1f}"
    print(
        f"{risc:8s} {z:22s} {len(d['tot']):4d}  {f(d['first'])}  {f(d['last'])}  {f(d['tot'])}  {statistics.median(d['n']):.0f}"
    )
