# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print OP CODE and DEVICE KERNEL DURATION [ns] of every row of an ops_perf_results CSV (no ttnn import)."""

import csv
import sys


def main(paths):
    for path in paths:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            dur = r.get("DEVICE KERNEL DURATION [ns]", "")
            if dur:
                print(
                    f"{path.split('/')[-2]}  {r.get('OP CODE', '?'):40s} {dur:>12s} ns  cores={r.get('CORE COUNT', '?')}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
