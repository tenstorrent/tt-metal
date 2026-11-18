"""Per-core zone sums (us) from profile_log_device.csv for the run with the most rows (the op of interest).
usage: r16_zones.py <profile_log_device.csv> [risc filter] [zones comma]"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
risc_f = sys.argv[2] if len(sys.argv) > 2 else ""
zones_f = sys.argv[3].split(",") if len(sys.argv) > 3 else None
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
open_, dur = {}, collections.defaultdict(int)
cnt = collections.Counter()
for r in by_run[run]:
    core, risc, z, typ, t = (int(r[1]), int(r[2])), r[3].strip(), r[10].strip(), r[11].strip(), int(r[5])
    if risc_f and risc != risc_f:
        continue
    key = (core, risc, z)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(t)
    elif typ == "ZONE_END" and open_.get(key):
        dur[key] += t - open_[key].pop()
        cnt[key] += 1
zones = sorted({k[2] for k in dur if not zones_f or k[2] in zones_f})
cores = sorted({(k[0], k[1]) for k in dur})
print("core risc " + " ".join(f"{z[:14]:>14s}" for z in zones))
for c, risc in cores:
    print(f"{c} {risc} " + " ".join(f"{dur.get((c, risc, z), 0)/1350:9.1f}({cnt.get((c,risc,z),0):2d})" for z in zones))
