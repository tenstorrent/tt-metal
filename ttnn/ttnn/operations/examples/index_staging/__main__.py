# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure index_staging on YOUR shape / index distribution / cores.

    python -m ttnn.operations.examples.index_staging [--shape ROWS,W] [--width W]
                                                      [--dist sorted,shuffled]
                                                      [--cores N] [--iters K]
                                                      [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, in-process profiler, post-run reset) and prints the same ns/op
table (variant x distribution). Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_index_staging.py::test_index_staging_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.index_staging")
    ap.add_argument("--shape", default="8,512", help="ROWS,W (W = indices per row). Default 8,512.")
    ap.add_argument("--width", type=int, default=None, help="override just W (keeps ROWS from --shape).")
    ap.add_argument(
        "--dist",
        default="sorted,shuffled",
        help="comma list of index distributions to sweep {sorted,shuffled}. Default sorted,shuffled.",
    )
    ap.add_argument("--cores", type=int, default=1, help="cores running the pipeline (each independent). Default 1.")
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="in-kernel repeat of the row range (K). 1=latency, large=steady. Default 1.",
    )
    ap.add_argument("--trials", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    shape = args.shape
    if args.width is not None:
        rows = shape.split(",")[0]
        shape = f"{rows},{args.width}"

    env = dict(
        os.environ,
        IS_SHAPE=shape,
        IS_DISTS=args.dist,
        IS_CORES=str(args.cores),
        IS_ITERS=str(args.iters),
        IS_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[index_staging] shape={shape} dist={args.dist} cores={args.cores} iters={args.iters} trials={args.trials}")
    print(f"[index_staging] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
