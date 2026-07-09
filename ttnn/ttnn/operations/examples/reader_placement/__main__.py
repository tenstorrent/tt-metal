# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure reader_placement on YOUR shapes/params.

    python -m ttnn.operations.examples.reader_placement [--shape H,W] [--cores 4,8]
                                                         [--block N] [--iters N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, Tracy profiler, post-run reset) and prints the same ns/op table.
Must be run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_reader_placement.py::test_reader_placement_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.reader_placement")
    ap.add_argument(
        "--shape", default="1024,2048", help="H,W of the bf16 tiled tensor (tile-aligned). Default 1024,2048."
    )
    ap.add_argument(
        "--cores", default="4,8", help="comma list of line lengths to sweep (<= min grid dim). Default 4,8."
    )
    ap.add_argument(
        "--block", type=int, default=16, help="pages issued per NoC barrier (in-flight pressure). Default 16."
    )
    ap.add_argument("--iters", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        RP_SHAPE=args.shape,
        RP_CORES=args.cores,
        RP_BLOCK=str(args.block),
        RP_ITERS=str(args.iters),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[reader_placement] shape={args.shape} cores={args.cores} block={args.block} iters={args.iters}")
    print(f"[reader_placement] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
