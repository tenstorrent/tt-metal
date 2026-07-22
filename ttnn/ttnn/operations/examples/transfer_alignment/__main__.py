# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure transfer_alignment on YOUR span size / count / cores.

    python -m ttnn.operations.examples.transfer_alignment [--width BYTES] [--spans N]
                                                           [--align {aligned,misaligned}]
                                                           [--variant {all,aligned,misaligned}]
                                                           [--cores N] [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path (device
lock, in-process profiler, post-run reset) and prints the same ns/op table
(variant x span-width x span-count). Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_transfer_alignment.py::test_transfer_alignment_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.transfer_alignment")
    ap.add_argument("--width", default=None, help="span bytes; comma list to sweep. Default 64,256,1024,4096.")
    ap.add_argument("--spans", default=None, help="number of spans/rows N; comma list to sweep. Default 16,64,256.")
    ap.add_argument(
        "--align",
        choices=["aligned", "misaligned"],
        default=None,
        help="run just one variant (alias for --variant). Default: both.",
    )
    ap.add_argument(
        "--variant",
        choices=["all", "aligned", "misaligned"],
        default="all",
        help="which method(s) to run/compare. Default all.",
    )
    ap.add_argument("--cores", type=int, default=1, help="cores running the pipeline (each independent). Default 1.")
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="in-kernel repeat of the row range (K). 1=latency, large=steady. Default 1.",
    )
    ap.add_argument("--trials", type=int, default=10, help="profiled launches per case (averaged). Default 10.")
    args = ap.parse_args()

    variant = args.align if args.align is not None else args.variant

    env = dict(os.environ, TA_CORES=str(args.cores), TA_ITERS=str(args.iters), TA_TRIALS=str(args.trials))
    if args.width is not None:
        env["TA_WIDTHS"] = args.width
    if args.spans is not None:
        env["TA_SPANS"] = args.spans
    if variant != "all":
        env["TA_VARIANTS"] = variant

    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[transfer_alignment] width={args.width or 'default'} spans={args.spans or 'default'} "
        f"variant={variant} cores={args.cores} iters={args.iters} trials={args.trials}"
    )
    print(f"[transfer_alignment] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
