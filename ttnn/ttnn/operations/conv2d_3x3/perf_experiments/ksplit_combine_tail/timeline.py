# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core tail timeline of each profiled dispatch (no device, no ttnn import).

    python3 timeline.py <profile_log_device.csv> [--top N]

For every run (dispatch on >= 8 cores), times in us from the run's first *-KERNEL start:
  walk_end    end of compute_matmul_walk (MATH)
  wait        compute_combine_wait span on UNPACK (start..end)
  comb_end    end of compute_combine on PACK
  end         last *-KERNEL end on that core
and the run span (first kernel start .. last kernel end), the N latest-ending cores.
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report  # noqa: E402


def spans(events):
    out = collections.defaultdict(list)  # (core, risc, zone) -> [(start, end)]
    stacks = collections.defaultdict(list)
    for core, risc, t, zone, kind in sorted(events, key=lambda e: e[2]):
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks[key].append(t)
        elif kind == "ZONE_END" and stacks[key]:
            out[key].append((stacks[key].pop(), t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    freq, runs = zone_report.load(args.log)
    for rid in sorted(runs):
        ev = runs[rid]
        cores = {c for c, *_ in ev}
        if len(cores) < 8:
            continue
        sp = spans(ev)
        kstart = min(s for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for s, _ in v)
        kend = max(e for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for _, e in v)
        us = lambda t: (t - kstart) / freq  # noqa: E731
        rows = []
        for c in sorted(cores):

            def first(risc, zone, which):
                v = sp.get((c, risc, zone))
                return us(v[-1][which]) if v else float("nan")

            end = max(us(e) for (cc, r, z), v in sp.items() if cc == c and z.endswith("-KERNEL") for _, e in v)
            rows.append(
                dict(
                    core=c,
                    walk_end=first("TRISC_1", "compute_matmul_walk", 1),
                    wait_s=first("TRISC_0", "compute_combine_wait", 0),
                    wait_e=first("TRISC_0", "compute_combine_wait", 1),
                    comb_end=first("TRISC_2", "compute_combine", 1),
                    end=end,
                )
            )
        span = (kend - kstart) / freq
        print(f"=== run {rid}: {len(cores)} cores, span {span:.2f} us")
        med = lambda k: statistics.median(r[k] for r in rows if r[k] == r[k])  # noqa: E731
        print(
            f"  p50: walk_end {med('walk_end'):.2f}  wait {med('wait_s'):.2f}..{med('wait_e'):.2f}"
            f"  comb_end {med('comb_end'):.2f}  end {med('end'):.2f}"
        )
        for r in sorted(rows, key=lambda r: -r["end"])[: args.top]:
            print(
                f"  core {r['core']}: walk_end {r['walk_end']:.2f}  wait {r['wait_s']:.2f}..{r['wait_e']:.2f}"
                f"  comb_end {r['comb_end']:.2f}  end {r['end']:.2f}"
            )
        latest_walk = max(r["walk_end"] for r in rows if r["walk_end"] == r["walk_end"])
        print(f"  latest walk_end {latest_walk:.2f}")


if __name__ == "__main__":
    main()
