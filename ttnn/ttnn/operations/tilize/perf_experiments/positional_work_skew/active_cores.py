#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""M1-vs-M2 discriminator: ACTIVE-CORE COUNT OVER TIME on the tilize focus shape.

M1 (sibling's read): the wall is total_bytes / aggregate_bandwidth; the per-core
row gradient is only a DRAIN ORDER on a saturated shared resource. Then demand
stays ~flat until near the very end and redistributing work is a NULL.

M2: the wall carries an end-of-kernel DEMAND COLLAPSE — the fast rows retire
early, only stragglers remain, aggregate DRAM demand falls below what the memory
system could serve, and the tail is served at reduced parallelism. Then flattening
the finish times shortens the wall toward the mean.

The observable that separates them: integrate the number of cores with a
`*-KERNEL` zone still open, as a function of time, and ask what fraction of the
WALL is spent below (say) 75% / 50% / 25% of peak concurrency. Under M1 the
concurrency stays high until a short cliff; under M2 there is a long thin tail.

Reads either
  (a) a raw `profile_log_device.csv` (per-core ZONE_START/ZONE_END rows), or
  (b) the sibling's saved `percore_maps.txt` grid dump (START/END maps),
so the discriminator costs no device time when a map is already on disk.
"""
import argparse
import collections
import csv
import re
import sys


def from_maps_txt(path, section, risc):
    """[(start, end), ...] ns rel to run t0, from percore_map.py's grid dump."""
    txt = open(path).read()
    secs = re.split(r"#+ (\w+) #+\n", txt)
    # secs = [pre, name1, body1, name2, body2, ...]
    bodies = {secs[i]: secs[i + 1] for i in range(1, len(secs) - 1, 2)}
    body = bodies[section]
    out = {}
    for key, which in ((f"{risc} START ns (rel)", 0), (f"{risc} END ns (rel)", 1)):
        m = re.search(re.escape(key) + r"\n(.*?)(?:\n\n|\n  [A-Z])", body, re.S)
        assert m, f"{key} not found in {section}"
        for line in m.group(1).strip().splitlines():
            mm = re.match(r"\s*y=(\d+)\s+(.*)", line)
            if not mm:
                continue
            y = int(mm.group(1))
            for x, v in enumerate(mm.group(2).split()):
                out.setdefault((x, y), [None, None])[which] = float(v)
    return [tuple(v) for v in out.values() if None not in v]


def from_device_csv(path, risc, run=None):
    rows = list(csv.reader(open(path)))[2:]
    st = collections.defaultdict(lambda: [None, None])
    for r in rows:
        if len(r) < 12 or r[3].strip() != risc:
            continue
        if not r[10].strip().endswith("-KERNEL"):
            continue
        rn = int(r[7])
        if run is not None and rn != run:
            continue
        cur = st[(rn, int(r[1]), int(r[2]))]
        t = int(r[5])
        if r[11].strip() == "ZONE_START":
            cur[0] = t if cur[0] is None else min(cur[0], t)
        else:
            cur[1] = t if cur[1] is None else max(cur[1], t)
    runs = sorted({k[0] for k in st})
    run = runs[-1] if run is None else run
    sel = [v for k, v in st.items() if k[0] == run and None not in v]
    t0 = min(v[0] for v in sel)
    return [(v[0] - t0, v[1] - t0) for v in sel], run


def report(spans, label):
    n = len(spans)
    t_end = max(e for _, e in spans)
    edges = sorted({t for s in spans for t in s})
    # piecewise-constant active count
    segs = []
    for a, b in zip(edges, edges[1:]):
        active = sum(1 for s, e in spans if s <= a and e > a)
        segs.append((a, b, active))
    total = sum((b - a) * c for a, b, c in segs)
    ideal_wall = total / n  # wall if every core were perfectly load balanced
    print(f"\n### {label}: {n} cores, wall {t_end:.0f} ns")
    print(f"  core-ns of work        = {total:.0f}   (mean per-core busy {total/n:.0f} ns)")
    print(f"  concurrency-flat wall  = {ideal_wall:.0f} ns  -> M2 upside {t_end/ideal_wall:.3f}x")
    for frac in (1.0, 0.9, 0.75, 0.5, 0.25):
        thr = frac * n
        t = sum((b - a) for a, b, c in segs if c < thr)
        print(f"  time with < {frac*100:5.1f}% cores active: {t:8.0f} ns  ({100*t/t_end:5.1f}% of wall)")
    # last-20%-of-wall mean concurrency
    cut = 0.8 * t_end
    tail = [(max(a, cut), b, c) for a, b, c in segs if b > cut]
    tail_mean = sum((b - a) * c for a, b, c in tail) / max(1e-9, sum(b - a for a, b, c in tail))
    head = [(a, min(b, cut), c) for a, b, c in segs if a < cut]
    head_mean = sum((b - a) * c for a, b, c in head) / max(1e-9, sum(b - a for a, b, c in head))
    print(f"  mean active cores: first 80% of wall = {head_mean:5.2f} | last 20% = {tail_mean:5.2f}")
    # coarse timeline
    print("  timeline (deciles of wall -> mean active cores):")
    line = []
    for i in range(10):
        a0, b0 = i * t_end / 10, (i + 1) * t_end / 10
        seg = [(max(a, a0), min(b, b0), c) for a, b, c in segs if b > a0 and a < b0]
        m = sum((b - a) * c for a, b, c in seg) / max(1e-9, sum(b - a for a, b, c in seg))
        line.append(f"{m:5.1f}")
    print("    " + " ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=None)
    ap.add_argument("--section", default="baseline_focus")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--riscs", default="NCRISC,BRISC")
    a = ap.parse_args()
    for risc in a.riscs.split(","):
        if a.maps:
            spans = from_maps_txt(a.maps, a.section, risc)
            report(spans, f"{a.section} {risc}")
        else:
            spans, run = from_device_csv(a.csv, risc, a.run)
            report(spans, f"run {run} {risc}")
    if a.maps:
        # union: a core is "active" if EITHER DM risc is inside its kernel zone
        allspans = []
        for risc in a.riscs.split(","):
            allspans += from_maps_txt(a.maps, a.section, risc)
        report(allspans, f"{a.section} NCRISC+BRISC union (DM demand proxy)")


if __name__ == "__main__":
    sys.exit(main())
