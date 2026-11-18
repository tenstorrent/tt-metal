# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION [ns] of every op row of ops_perf_results CSVs (no ttnn import).

    python3 parse_csv.py <ops_perf_results.csv> [...]
"""

import csv
import sys


def main(paths):
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                dur = r.get("DEVICE KERNEL DURATION [ns]", "")
                if dur:
                    print(f"{r.get('OP CODE', '?'):32s} {dur:>10s} ns  cores={r.get('CORE COUNT', '?')}")


if __name__ == "__main__":
    main(sys.argv[1:])
