#!/usr/bin/env python3
"""Per-core grid (logical y rows x x cols) of one zone metric, for a run: grid.py <device.csv> <run id> [p1|p2|ws|tot]
or grid.py --json <baseline_core_times.json> <case> [key]."""
import json, sys
from core_times import core_times

if sys.argv[1] == "--json":
    d = json.load(open(sys.argv[2]))[sys.argv[3]]
    key = sys.argv[4] if len(sys.argv) > 4 else "p1"
else:
    d = core_times(sys.argv[1], sys.argv[2])
    key = sys.argv[3] if len(sys.argv) > 3 else "p1"
vals = {}
for k, v in d.items():
    x, y = map(int, k.split(","))
    vals[(x, y)] = v.get("p1", 0) + v.get("p2", 0) if key == "tot" else v.get(key, 0)
xs = sorted({c[0] for c in vals})
ys = sorted({c[1] for c in vals})
print(f"{key}: max {max(vals.values()):.1f}  min {min(vals.values()):.1f}  mean {sum(vals.values())/len(vals):.1f}")
for y in ys:
    print(f"y{y:2d} " + " ".join(f"{vals.get((x, y), 0):7.1f}" for x in xs))
