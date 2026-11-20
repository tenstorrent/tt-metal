# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Median / range per (case, variant) of report.py lines on stdin. Host-only: python3 summ.py < report_lines"""
import collections
import statistics
import sys

d = collections.defaultdict(list)
for line in sys.stdin:
    p = line.split()
    if len(p) > 3 and p[3] == "ns":
        d[(p[0], p[1])].append(int(p[2]))
for c in sorted({k[0] for k in d}):
    b = statistics.median(d[(c, "baseline")]) if (c, "baseline") in d else None
    out = []
    for (cc, v), xs in d.items():
        if cc != c:
            continue
        m = statistics.median(xs)
        out.append(f"{v}={m:.0f}" + (f"({m / b - 1:+.1%})" if b else "") + f" {sorted(xs)}")
    print(c, "  ".join(out))
