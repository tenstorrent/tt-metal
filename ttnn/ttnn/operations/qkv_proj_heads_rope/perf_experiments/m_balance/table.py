"""reader_table zone per core (first op only): sum us / calls, bucketed by calls. table.py <csv>"""
import collections, csv, statistics, sys

with open(sys.argv[1]) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_, tot, cnt = {}, collections.defaultdict(float), collections.defaultdict(int)
t0 = None
for r in rows:
    if len(r) < 12:
        continue
    core = (int(r[1]), int(r[2]))
    t, zone, typ = int(r[5]), r[10].strip(), r[11].strip()
    t0 = t if t0 is None else t0
    if t - t0 > 200 * freq:
        continue
    key = (core, zone)
    if typ == "ZONE_START":
        open_[key] = t
    elif typ == "ZONE_END" and key in open_:
        tot[key] += t - open_.pop(key)
        cnt[key] += 1
by = collections.defaultdict(list)
for k in tot:
    if k[1] == "reader_table":
        by[cnt[k]].append(tot[k] / freq)
for c in sorted(by):
    print(f"calls {c}: cores {len(by[c])} sum med {statistics.median(by[c]):.1f} max {max(by[c]):.1f} us")
