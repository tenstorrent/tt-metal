"""Kernel span (us): min *-KERNEL start to max *-KERNEL end, per op (split by run id), from profile_log_device.csv."""
import collections, csv, sys

with open(sys.argv[1]) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
ops = collections.defaultdict(lambda: [None, None])
for r in rows:
    if len(r) < 12 or not r[10].strip().endswith("-KERNEL"):
        continue
    run, t, typ = r[7].strip(), int(r[5]), r[11].strip()
    span = ops[run]
    if typ == "ZONE_START":
        span[0] = t if span[0] is None else min(span[0], t)
    elif typ == "ZONE_END":
        span[1] = t if span[1] is None else max(span[1], t)
print("SPAN_US", " ".join(f"{(e - s) / freq:.1f}" for s, e in ops.values() if s and e))
