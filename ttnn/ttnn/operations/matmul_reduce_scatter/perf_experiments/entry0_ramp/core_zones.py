"""Per-core zone list in a time window: python core_zones.py <device.csv> <device> <replay 0..2> <t0_us> <t1_us> [zone-regex]"""
import csv, re, sys
from collections import defaultdict

path, dev, rep, t0w, t1w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
rx = re.compile(sys.argv[6]) if len(sys.argv) > 6 else None
f = open(path)
first = f.readline()
freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) if "CHIP_FREQ" in first else 1350.0
rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
rs = [[c.strip() for c in r] for r in rows[1:] if int(r[ix["PCIe slot"]]) == dev]
keyf = lambda r: (r[ix["run host ID"]], r[ix["trace id counter"]])
keys = []
for r in rs:
    if keyf(r) not in keys:
        keys.append(keyf(r))
key = keys[-3 + rep]
rs = [r for r in rs if keyf(r) == key]
T0 = min(int(r[ix["time[cycles since reset]"]]) for r in rs)
open_ = {}
out = defaultdict(list)
for r in rs:
    if r[ix["type"]] not in ("ZONE_START", "ZONE_END"):
        continue
    k = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]], r[ix["zone name"]])
    t = (int(r[ix["time[cycles since reset]"]]) - T0) / freq
    if r[ix["type"]] == "ZONE_START":
        open_.setdefault(k, []).append(t)
    elif open_.get(k):
        s = open_[k].pop()
        if e_ok := (t >= t0w and s <= t1w) and (rx is None or rx.search(k[3])):
            out[(k[0], k[1], k[2])].append((s, t, k[3]))
for c in sorted(out):
    print(c, "  ".join(f"{z}[{s:.1f}-{e:.1f}]" for s, e, z in sorted(out[c])))
