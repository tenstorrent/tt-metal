# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the four read-coalescing × bank-affinity combinations on width-sharded DRAM."""

import argparse
import os
import subprocess
import tempfile
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_read_coalescing.py::test_read_coalescing_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.read_coalescing")
    ap.add_argument(
        "--cases",
        default="auto",
        help="comma list of HxWxSHARD_WIDTH (width-sharded DRAM, shard width in elements, one shard per bank), "
        "or auto (default): cases sized to the grid.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="all, or a comma list of none,coalesce,affinity,both. Default all.",
    )
    ap.add_argument("--kernel-iters", type=int, default=1, help="in-kernel repeat. 1 = per-launch latency. Default 1.")
    ap.add_argument("--trials", type=int, default=5, help="trials per variant (median reported). Default 5.")
    ap.add_argument("--launches", type=int, default=10, help="launches averaged per trial. Default 10.")
    ap.add_argument("--report", default=None, help="also write the table to this file.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        RC_CASES=args.cases,
        RC_VARIANT=args.variant,
        RC_ITERS=str(args.kernel_iters),
        RC_TRIALS=str(args.trials),
        RC_LAUNCHES=str(args.launches),
    )
    # pytest captures the logger, so the table always goes through a file and is printed here.
    if args.trials < 1 or args.launches < 1 or args.kernel_iters < 1:
        ap.error("trials, launches and kernel-iters must be positive")
    if args.report:
        report = os.path.abspath(args.report)
    else:
        fd, report = tempfile.mkstemp(prefix="read_coalescing_", suffix=".txt")
        os.close(fd)
    env["RC_REPORT"] = report
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[read_coalescing] cases={args.cases} variant={args.variant} kernel_iters={args.kernel_iters}")
    print(f"[read_coalescing] {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, stdout=subprocess.DEVNULL)
    if os.path.exists(report):
        print(open(report).read())
        if not args.report:
            os.remove(report)
    return rc


if __name__ == "__main__":
    sys.exit(main())
