# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core startup / end summary of the profiled dispatches in a profile_log_device.csv (no device).

    python3 hk_cores.py <profile_log_device.csv> [--top N]

Per core: TRISC_1 kernel end, the first and second compute_matmul_walk starts, the first writer_weight_fetch end
(weight round 0 landed), all in us from the dispatch's first marker."""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    freq, runs = zone_report.load(args.log)
    for rid in sorted(runs):
        ev = runs[rid]
        if len({c for c, *_ in ev}) < 8:
            continue
        t0 = min(e[2] for e in ev)
        st = collections.defaultdict(list)
        spans = collections.defaultdict(lambda: collections.defaultdict(list))
        for c, r, t, z, k in sorted(ev, key=lambda e: e[2]):
            key = (c, r, z)
            if k == "ZONE_START":
                st[key].append(t)
            elif st[key]:
                s = st[key].pop()
                spans[c][(r, z)].append(((s - t0) / freq, (t - t0) / freq))
        rows = []
        for c, d in spans.items():
            ke = d[("TRISC_1", "TRISC-KERNEL")]
            walks = d[("TRISC_1", "compute_matmul_walk")] or d[("TRISC_1", "compute_matmul_stream")]
            wf = d[("BRISC", "writer_weight_fetch")]
            rows.append(
                (
                    ke[0][1] if ke else float("nan"),
                    c,
                    walks[0][0] if walks else float("nan"),
                    walks[1][0] if len(walks) > 1 else float("nan"),
                    wf[0][1] if wf else float("nan"),
                )
            )
        rows.sort(reverse=True)
        print(f"=== run {rid}: {len(rows)} cores")
        for e in rows[: args.top]:
            print(f"  end {e[0]:7.2f} core {e[1]}  walk0 {e[2]:7.2f}  walk1 {e[3]:7.2f}  w0 landed {e[4]:7.2f}")
        for name, i in (("end", 0), ("walk0 start", 2), ("walk1 start", 3), ("w0 landed", 4)):
            vals = [e[i] for e in rows if e[i] == e[i]]
            if vals:
                print(f"  {name:<12} p50 {statistics.median(vals):7.2f}  max {max(vals):7.2f}  min {min(vals):7.2f}")


if __name__ == "__main__":
    main()
