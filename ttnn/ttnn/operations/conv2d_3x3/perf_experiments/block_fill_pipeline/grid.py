# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grid of one zone's last end (or first start) per core for one dispatch: python3 grid.py <log> <dispatch> <zone> [start]"""
import sys
import timeline

freq, runs = timeline.load(sys.argv[1])
ids = sorted(rid for rid, ev in runs.items() if len({c for c, *_ in ev}) >= 8)
first, last, kend = timeline.spans(runs[ids[int(sys.argv[2])]], freq)
zone = sys.argv[3]
src = first if len(sys.argv) > 4 else last
cores = sorted({c for c, z in src if z == zone})
xs = sorted({c[0] for c in cores})
ys = sorted({c[1] for c in cores})
print("y\\x " + " ".join(f"{x:>6}" for x in xs))
for y in ys:
    print(f"{y:>3} " + " ".join(f"{src.get(((x, y), zone), float('nan')):6.1f}" for x in xs))
