# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure the accumulate+SFPU-finalize reduce fast path over a 2-D (Ht, Wt, NC) block.

    python -m ttnn.operations.examples.reduce_block [options]

Options:
    --dim     {all|row|col|scalar} ...   reduce dimension(s)
    --variant {all|reduce_tile|accumulate_via_add|accumulate_via_add_inline} ...   which path(s) to compare
    --shape   Ht Wt NC                   a block to reduce (repeatable); overrides the built-in sweep,
                                         applied to every selected dim (a shape is valid for any dim)
    --trials  N                          timed passes; median +/- std
    --kernel-iters K                     in-kernel loop count (K large = steady-state)
    --report  PATH                       write the report to PATH
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import DIMS

# The perf comparison runs the three concrete variants (the `dispatch` variant just resolves to one of
# `reduce_tile` / `accumulate_via_add`, so it adds no new measurement here).
_PERF_VARIANTS = ("reduce_tile", "accumulate_via_add", "accumulate_via_add_inline")
_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_reduce_block.py::test_reduce_block_device_perf"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.reduce_block")
    parser.add_argument("--dim", nargs="+", choices=("all",) + DIMS, default=["all"])
    parser.add_argument("--variant", nargs="+", choices=("all",) + _PERF_VARIANTS, default=["all"])
    parser.add_argument("--shape", nargs=3, type=_positive, action="append", metavar=("Ht", "Wt", "NC"))
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=200)
    parser.add_argument("--report")
    args = parser.parse_args()

    def expand(sel, allowed):
        return list(allowed) if "all" in sel else list(dict.fromkeys(sel))

    env = dict(
        os.environ,
        REDBLK_DIMS=",".join(expand(args.dim, DIMS)),
        REDBLK_VARIANTS=",".join(expand(args.variant, _PERF_VARIANTS)),
        REDBLK_TRIALS=str(args.trials),
        REDBLK_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.shape:
        env["REDBLK_SHAPES"] = ";".join(f"{ht},{wt},{nc}" for ht, wt, nc in args.shape)
    if args.report:
        env["REDBLK_REPORT"] = args.report

    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST_NODE], env=env)


if __name__ == "__main__":
    sys.exit(main())
