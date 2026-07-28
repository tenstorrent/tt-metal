# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure constant_synthesis on YOUR output shape / constant / cores.

    python -m ttnn.operations.examples.constant_synthesis [--shape ROWS,W] [--value V]
                                                          [--variant all|stream_from_dram|synthesize]
                                                          [--cores N] [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, in-process profiler, post-run reset) and prints the same ns/op
table (variant x cores). Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_constant_synthesis.py::test_constant_synthesis_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.constant_synthesis")
    ap.add_argument("--shape", default="4096,1024", help="ROWS,W of the constant-valued output. Default 4096,1024.")
    ap.add_argument("--value", type=float, default=1.0, help="the constant that fills the output. Default 1.0.")
    ap.add_argument(
        "--variant",
        default="all",
        help="which method(s) to run/compare {all,stream_from_dram,synthesize}. Default all.",
    )
    ap.add_argument(
        "--cores",
        default=None,
        help="comma list of core counts to sweep (e.g. 1,64). Default: 1 and the full grid.",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="in-kernel repeat of the page range (K). 1=latency, large=steady. Default 1.",
    )
    ap.add_argument(
        "--block",
        type=int,
        default=8,
        help="async reads/writes in flight per NoC barrier (cb_data = 2*block deep). Default 8.",
    )
    ap.add_argument("--trials", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        CS_SHAPE=args.shape,
        CS_VALUE=str(args.value),
        CS_VARIANT=args.variant,
        CS_ITERS=str(args.iters),
        CS_BLOCK=str(args.block),
        CS_TRIALS=str(args.trials),
    )
    if args.cores is not None:
        env["CS_CORES"] = args.cores
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[constant_synthesis] shape={args.shape} value={args.value} variant={args.variant} "
        f"cores={args.cores or '1,full'} iters={args.iters} trials={args.trials}"
    )
    print(f"[constant_synthesis] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
