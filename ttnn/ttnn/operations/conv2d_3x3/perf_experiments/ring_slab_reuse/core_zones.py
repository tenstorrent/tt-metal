# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core zone sums of each dispatch in a profile_log_device.csv (no device): the slowest cores by kernel span.

    python3 core_zones.py <profile_log_device.csv> [--top N] [--zones a,b]
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--zones", default="compute_untilize,compute_matmul_walk,reader_fill,reader_gather,reader_center")
    args = ap.parse_args()
    freq, runs = zone_report.load(args.log)
    zones = args.zones.split(",")
    for rid in sorted(runs):
        ev = runs[rid]
        if len({c for c, *_ in ev}) < 8:
            continue
        span = {}
        t0 = min(t for _, _, t, _, _ in ev)
        stacks = collections.defaultdict(list)
        sums = collections.defaultdict(float)
        for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
            key = (core, risc, zone)
            if kind == "ZONE_START":
                stacks[key].append(t)
            elif kind == "ZONE_END" and stacks[key]:
                start = stacks[key].pop()
                sums[key] += t - start
                if zone.endswith("-KERNEL") and risc == "TRISC_2":
                    span[core] = ((start - t0) / freq, (t - t0) / freq)
        print(f"=== run {rid}: {len(span)} cores; kernel end max {max(e for _, e in span.values()):.1f} us")
        for core, (s, e) in sorted(span.items(), key=lambda kv: -kv[1][1])[: args.top]:
            parts = []
            for z in zones:
                for risc in ("TRISC_2", "NCRISC", "BRISC"):
                    v = sums.get((core, risc, z))
                    if v:
                        parts.append(f"{z}@{risc[:2]} {v / freq:6.1f}")
                        break
            print(f"  core {core} start {s:6.1f} end {e:6.1f}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
