# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per run: distribution over cores of the prelude's end (us from the run's first marker) and of the kernel end.

    python3 prelude_ends.py <profile_log_device.csv>
"""

import statistics
import sys

from timeline import spans_of, zr


def main():
    freq, runs = zr.load(sys.argv[1])
    for run_id in sorted(runs):
        sp = spans_of(runs[run_id], freq)
        pre, gat, kend, walk_start = [], [], [], []
        for core, spans in sp.items():
            p = [e for s, e, r, z in spans if r == "TRISC_2" and z in ("compute_prelude", "compute_tilize")]
            g = [e for s, e, r, z in spans if r == "NCRISC" and z == "reader_gather"]
            k = [e for s, e, r, z in spans if r == "TRISC_2" and z == "TRISC-KERNEL"]
            if p and g and k:
                pre.append(max(p))
                gat.append(max(g))
                kend.append(max(k))
        q = lambda xs: f"p50 {statistics.median(xs):6.2f} max {max(xs):6.2f} min {min(xs):6.2f}"
        print(f"run {run_id:>6} cores {len(pre):3}  prelude_end {q(pre)}  gather_end {q(gat)}  kernel_end {q(kend)}")


if __name__ == "__main__":
    main()
