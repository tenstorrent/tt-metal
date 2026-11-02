"""Fill timeline: per (RISC, zone), median / max over cores of the FIRST occurrence's start and end
(us from the earliest kernel start). python first.py <profile_log_device.csv>"""
import collections, csv, statistics, sys

with open(sys.argv[1]) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_, first = {}, {}
t0 = None
for r in rows:
    if len(r) < 12:
        continue
    core, risc, t, zone, typ = (r[1].strip(), r[2].strip()), r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    key = (core, risc, zone)
    if typ == "ZONE_START" and key not in first:
        open_[key] = t
    elif typ == "ZONE_END" and key in open_ and key not in first:
        first[key] = (open_.pop(key), t)
by = collections.defaultdict(list)
for (core, risc, zone), (s, e) in first.items():
    by[(risc, zone)].append(((s - t0) / freq, (e - t0) / freq))
for (risc, zone), v in sorted(by.items(), key=lambda kv: (kv[0][0], statistics.median(x[0] for x in kv[1]))):
    if zone.endswith("-FW"):
        continue
    print(
        f"{risc:7} {zone:26} {len(v):4d} start_med {statistics.median(x[0] for x in v):6.2f} "
        f"end_med {statistics.median(x[1] for x in v):6.2f} end_max {max(x[1] for x in v):6.2f}"
    )
