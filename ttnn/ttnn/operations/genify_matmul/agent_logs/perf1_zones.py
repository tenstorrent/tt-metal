import csv, sys, collections, statistics as st

# per (core,risc,zone) -> list of durations; picks the LAST run host id (the op of interest = longest program)
f = sys.argv[1]
rows = list(csv.reader(open(f)))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
# pick run with the most rows
run = max(by_run, key=lambda k: len(by_run[k]))
open_ = {}
dur = collections.defaultdict(list)
markers = collections.Counter()
for r in by_run[run]:
    core = (int(r[1]), int(r[2]))
    risc = r[3].strip()
    z = r[10].strip()
    typ = r[11].strip()
    t = int(r[5])
    markers[(core, risc)] += 1
    key = (core, risc, z)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(t)
    elif typ == "ZONE_END" and open_.get(key):
        dur[key].append(t - open_[key].pop())
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for (core, risc, z), ds in dur.items():
    agg[(risc, z)]["sum"].append(sum(ds))
    agg[(risc, z)]["n"].append(len(ds))
print(f"run {run}; max markers per (core,risc) = {max(markers.values())}")
print(
    f"{'risc':8s} {'zone':22s} {'cores':>5s} {'exec':>5s} {'p50 sum us':>10s} {'max sum us':>10s} {'min sum us':>10s}"
)
for (risc, z), d in sorted(agg.items()):
    s = d["sum"]
    print(
        f"{risc:8s} {z:22s} {len(s):5d} {int(st.median(d['n'])):5d} {st.median(s)/1350:10.1f} {max(s)/1350:10.1f} {min(s)/1350:10.1f}"
    )
