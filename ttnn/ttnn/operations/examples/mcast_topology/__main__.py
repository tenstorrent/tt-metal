# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure work-split topology (and the multicast shape it forces) on YOUR matmul shape.

    python -m ttnn.operations.examples.mcast_topology [--mt 8] [--nt 32] [--kt 4] [--trials 5]

Runs the device-perf test through scripts/run_safe_pytest.sh (device lock, in-process profiler,
post-run reset) and prints the per-variant core occupancy + device kernel duration. Run from the
repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_mcast_topology.py::test_mcast_topology_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.mcast_topology")
    ap.add_argument("--mt", type=int, default=8, help="output tile-rows (M in tiles). Default 8.")
    ap.add_argument("--nt", type=int, default=32, help="output tile-cols (N in tiles). Default 32.")
    ap.add_argument("--kt", type=int, default=4, help="contraction tiles (K in tiles). Default 4.")
    ap.add_argument("--trials", type=int, default=5, help="profiled rounds (median). Default 5.")
    ap.add_argument(
        "--variant",
        default="all",
        help="which method(s) to run: all | per_core_dram | mcast_1d_pair (comma-separated). Default all.",
    )
    args = ap.parse_args()

    env = dict(
        os.environ,
        MCT_MT=str(args.mt),
        MCT_NT=str(args.nt),
        MCT_KT=str(args.kt),
        MCT_TRIALS=str(args.trials),
        MCT_VARIANTS=args.variant,
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[mcast_topology] M={args.mt}t N={args.nt}t K={args.kt}t trials={args.trials} variant={args.variant}")
    print(f"[mcast_topology] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
