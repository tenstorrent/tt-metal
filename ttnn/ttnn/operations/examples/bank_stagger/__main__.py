# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure the bank-de-clustered issue order on YOUR shapes.

    python -m ttnn.operations.examples.bank_stagger [--cases 32x8192x4,32x16384x8]
                                                     [--variant all|none,read,blocks,write,combined]
                                                     [--kernel-iters K] [--trials N] [--report PATH]

Each case is HxWxCHUNK: a bf16 ROW_MAJOR [H, W] tensor tilized with CHUNK tile-columns
per work unit (read size = CHUNK * 64 B per row). Runs the device-perf test through
scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run reset) and
prints ns per variant plus the ratio vs `none`. Run from the repo root with the
Python env active.
"""

import argparse
import os
import subprocess
import tempfile
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py::test_bank_stagger_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.bank_stagger")
    ap.add_argument(
        "--cases",
        default="auto",
        help="comma list of HxWxCHUNK (interleaved) or HxWxCHUNKws (width-sharded, one shard per DRAM bank), or auto (default): cases sized to the grid.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="all, or a comma list of none,read,blocks,write,combined. Default all.",
    )
    ap.add_argument("--kernel-iters", type=int, default=1, help="in-kernel repeat. 1 = per-launch latency. Default 1.")
    ap.add_argument("--trials", type=int, default=5, help="trials per variant (median reported). Default 5.")
    ap.add_argument("--launches", type=int, default=10, help="launches averaged per trial. Default 10.")
    ap.add_argument("--report", default=None, help="also write the table to this file.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        BS_CASES=args.cases,
        BS_VARIANT=args.variant,
        BS_ITERS=str(args.kernel_iters),
        BS_TRIALS=str(args.trials),
        BS_LAUNCHES=str(args.launches),
    )
    # pytest captures the logger, so the table always goes through a file and is printed here.
    report = os.path.abspath(args.report) if args.report else tempfile.mktemp(prefix="bank_stagger_", suffix=".txt")
    env["BS_REPORT"] = report
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[bank_stagger] cases={args.cases} variant={args.variant} kernel_iters={args.kernel_iters}")
    print(f"[bank_stagger] {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, stdout=subprocess.DEVNULL)
    if os.path.exists(report):
        print(open(report).read())
        if not args.report:
            os.remove(report)
    return rc


if __name__ == "__main__":
    sys.exit(main())
