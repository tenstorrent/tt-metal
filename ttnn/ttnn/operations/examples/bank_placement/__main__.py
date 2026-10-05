# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure bank-aware core placement on YOUR sizes.

    python -m ttnn.operations.examples.bank_placement [--cases 3072x1024,3072x4096]
                                                       [--pattern all|affine|spread]
                                                       [--variant all|row_major,bank_near,bank_shuffled]
                                                       [--block B] [--kernel-iters K] [--trials N] [--report PATH]

Each case is ROWSxWIDTH: a bf16 ROW_MAJOR [ROWS, WIDTH] tensor, one row == one DRAM page
of WIDTH*2 bytes; ROWS must be a multiple of the DRAM bank count. Runs the device-perf
test through scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run
reset) and prints ns, GB/s and the ratio vs `row_major`. Run from the repo root with
the Python env active.
"""

import argparse
import os
import subprocess
import sys
import tempfile

_TEST = "tests/ttnn/unit_tests/operations/examples/test_bank_placement.py::test_bank_placement_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.bank_placement")
    ap.add_argument("--cases", default="3072x256,3072x1024,3072x4096", help="comma list of ROWSxWIDTH (bf16).")
    ap.add_argument("--pattern", default="all", help="all, affine (one bank per core) or spread (contiguous runs).")
    ap.add_argument("--variant", default="all", help="all, or a comma list of row_major,bank_near,bank_shuffled.")
    ap.add_argument("--block", type=int, default=8, help="pages per NoC barrier. Default 8.")
    ap.add_argument("--kernel-iters", type=int, default=1, help="in-kernel repeat. 1 = per-launch latency. Default 1.")
    ap.add_argument("--trials", type=int, default=5, help="trials per variant (median reported). Default 5.")
    ap.add_argument("--launches", type=int, default=10, help="launches averaged per trial. Default 10.")
    ap.add_argument("--report", default=None, help="also write the table to this file.")
    args = ap.parse_args()

    # pytest captures the logger, so the table always goes through a file and is printed here.
    report = os.path.abspath(args.report) if args.report else tempfile.mktemp(prefix="bank_placement_", suffix=".txt")
    env = dict(
        os.environ,
        BP_CASES=args.cases,
        BP_PATTERN=args.pattern,
        BP_VARIANT=args.variant,
        BP_BLOCK=str(args.block),
        BP_ITERS=str(args.kernel_iters),
        BP_TRIALS=str(args.trials),
        BP_LAUNCHES=str(args.launches),
        BP_REPORT=report,
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[bank_placement] cases={args.cases} pattern={args.pattern} variant={args.variant} block={args.block}")
    print(f"[bank_placement] {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, stdout=subprocess.DEVNULL)
    if os.path.exists(report):
        print(open(report).read())
        if not args.report:
            os.remove(report)
    return rc


if __name__ == "__main__":
    sys.exit(main())
