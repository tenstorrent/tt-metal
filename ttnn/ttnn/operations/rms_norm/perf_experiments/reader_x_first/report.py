# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: print the per-op rows of a `--profile` ops_perf_results CSV in execution order, labelled
with the reader ordering names (rows of one regime run map 1:1 onto that regime's ORDERS tuple).

    python3 report.py <ops_perf_results.csv> [order_names...]
"""

import csv
import sys

ORDER_NAMES = [
    "baseline",
    "xfirst_one_barrier",
    "xfirst_split_barriers",
    "xfirst_trid_barriers",
    "scaler_x_gamma",
    "scaler_xg_one_barrier",
    "scaler_xg_trid",
    "x_gamma_scaler_last",
    "x_scaler_gamma",
]


def main():
    path = sys.argv[1]
    names = sys.argv[2:] or ORDER_NAMES
    with open(path) as f:
        rows = list(csv.DictReader(f))
    base = None
    for i, r in enumerate(rows):
        dur = int(r["DEVICE KERNEL DURATION [ns]"])
        if base is None:
            base = dur
        name = names[i] if i < len(names) else f"row{i}"
        print(f"{name:24s} cores={r['CORE COUNT']:>4s} kernel_ns={dur:7d}  baseline/this={base/dur:5.2f}")


if __name__ == "__main__":
    main()
