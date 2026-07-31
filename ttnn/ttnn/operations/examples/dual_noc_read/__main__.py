# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure dual_noc_read on YOUR shape / reads-per-barrier.

    python -m ttnn.operations.examples.dual_noc_read [--shape H,W]
                                                     [--blocks 1,2,4,8,16,32]
                                                     [--variant all|two_riscv|two_riscv_sem]
                                                     [--trials N] [--iters N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path (device lock,
in-process profiler, post-run reset) and prints the same two tables: the full op (C = A*B) and the
payload-ablated pure-read ceiling, for every block x variant. Run from the repo root with the
Python env active.

Note there is no --kernel-iters flag: the output is L1-resident and no kernel drains it, so a launch
performs exactly one pass over the tiles. Amortize launch overhead with --shape (more tiles) instead.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.dual_noc_read")
    ap.add_argument(
        "--shape",
        default="1024,128",
        help="H,W of each bf16 tiled operand (tile-aligned; total tiles must divide every --blocks "
        "value). Default 1024,128 = 128 tiles/operand.",
    )
    ap.add_argument(
        "--blocks",
        default="1,2,4,8,16,32",
        help="comma list of reads-per-barrier (per RISC) to sweep. Default 1,2,4,8,16,32.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="all | comma list from two_riscv,two_riscv_sem. The one_riscv baseline is always "
        "measured so speedups have a reference. Default all.",
    )
    ap.add_argument("--trials", type=int, default=5, help="independent profiler windows; median + spread. Default 5.")
    ap.add_argument("--iters", type=int, default=10, help="launches averaged inside each window. Default 10.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        DNR_SHAPE=args.shape,
        DNR_BLOCKS=args.blocks,
        DNR_TRIALS=str(args.trials),
        DNR_ITERS=str(args.iters),
    )
    if args.variant != "all":
        env["DNR_VARIANTS"] = args.variant

    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[dual_noc_read] shape={args.shape} blocks={args.blocks} variant={args.variant} "
        f"trials={args.trials} iters={args.iters}"
    )
    print(f"[dual_noc_read] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
