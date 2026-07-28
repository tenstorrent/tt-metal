# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure page_walk_order on YOUR page count / page size / strides.

    python -m ttnn.operations.examples.page_walk_order [--pages N] [--page-size ELEMS]
                                                        [--strides auto|s0,s1,...]
                                                        [--variant all|<name>...]
                                                        [--block B] [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path (device
lock, in-process profiler, post-run reset) and prints the same ns/op + read GB/s table.
Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_page_walk_order.py::test_page_walk_order_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.page_walk_order")
    ap.add_argument(
        "--pages",
        type=int,
        default=1536,
        help="requested page count N (rounded up to a multiple of the queried bank count). Default 1536.",
    )
    ap.add_argument(
        "--page-size",
        type=int,
        default=1024,
        help="elements (bf16) per page; page bytes = 2*this. Multiple of 16. Default 1024 (2 KB).",
    )
    ap.add_argument(
        "--strides",
        default="auto",
        help="'auto' = the named variants' strides from the bank count, or an explicit comma list of integer strides. Default auto.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="which named walk order(s) to run when --strides auto: all|bank_stride|unit_stride|coprime_stride (comma list). Default all.",
    )
    ap.add_argument("--block", type=int, default=0, help="reads issued per barrier (0 = auto = 2*banks). Default 0.")
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="in-kernel repeat of the full walk (K). 1=latency, large=steady. Default 1.",
    )
    ap.add_argument("--trials", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        PWO_PAGES=str(args.pages),
        PWO_PAGE_SIZE=str(args.page_size),
        PWO_STRIDES=args.strides,
        PWO_VARIANT=args.variant,
        PWO_BLOCK=str(args.block),
        PWO_ITERS=str(args.iters),
        PWO_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[page_walk_order] pages={args.pages} page_size={args.page_size} strides={args.strides} "
        f"variant={args.variant} block={args.block} iters={args.iters} trials={args.trials}"
    )
    print(f"[page_walk_order] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
