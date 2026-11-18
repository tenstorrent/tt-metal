# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One core's prelude zone timeline (no device): python3 tl.py <device.csv[.gz]> <run index 0..> <x,y> [all]"""
import collections
import gzip
import io
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import zone_report as zr

path, idx, core = sys.argv[1], int(sys.argv[2]), tuple(int(x) for x in sys.argv[3].split(","))
show_all = len(sys.argv) > 4
if path.endswith(".gz"):
    tmp = "/tmp/nct_tl.csv"
    with gzip.open(path, "rt") as f, open(tmp, "w") as g:
        g.write(f.read())
    path = tmp
freq, runs = zr.load(path)
run = sorted(runs)[idx]
ev = [e for e in runs[run] if e[0] == core]
t0 = min(e[2] for e in ev)
stack = collections.defaultdict(list)
rows = []
for c, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
    if kind == "ZONE_START":
        stack[(risc, zone)].append(t)
    elif kind == "ZONE_END" and stack[(risc, zone)]:
        s = stack[(risc, zone)].pop()
        rows.append(((s - t0) / freq, (t - t0) / freq, risc, zone))
for s, e, risc, zone in sorted(rows):
    if risc in ("TRISC_0", "TRISC_2"):
        continue
    if not show_all and ("ring_" in zone or ("weight_fetch" in zone and s > 15)):
        continue
    print(f"{s:7.2f} {e:7.2f} {e-s:6.2f} {risc:8s} {zone}")
