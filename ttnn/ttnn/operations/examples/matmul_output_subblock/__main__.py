# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure matmul output-subblock sizing on YOUR matmul shape.

    python -m ttnn.operations.examples.matmul_output_subblock [--mt 16] [--nt 16]
                                                              [--kt-sweep 1,2,4,8]
                                                              [--iters 50] [--trials 10]

Runs the device-perf test through scripts/run_safe_pytest.sh (device lock, in-process
profiler, post-run reset) and prints the output-subblock x Kt table (+ a FMAs-removed
microbench). Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_device_perf"
)


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.matmul_output_subblock")
    ap.add_argument("--mt", type=int, default=16, help="output tile rows (M in tiles). Default 16.")
    ap.add_argument("--nt", type=int, default=16, help="output tile cols (N in tiles). Default 16.")
    ap.add_argument("--kt-sweep", default="1,2,4,8", help="comma list of contraction depths Kt. Default 1,2,4,8.")
    ap.add_argument("--iters", type=int, default=50, help="in-kernel repeat (steady-state). Default 50.")
    ap.add_argument("--trials", type=int, default=10, help="profiled rounds (median). Default 10.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        MOS_MT=str(args.mt),
        MOS_NT=str(args.nt),
        MOS_KT_SWEEP=args.kt_sweep,
        MOS_ITERS=str(args.iters),
        MOS_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[matmul_output_subblock] M={args.mt}t N={args.nt}t Kt-sweep={args.kt_sweep} iters={args.iters}")
    print(f"[matmul_output_subblock] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
