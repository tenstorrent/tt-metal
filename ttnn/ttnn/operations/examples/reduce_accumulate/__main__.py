# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure the accumulate+SFPU-finalize reduce fast path vs the reduce library.

    python -m ttnn.operations.examples.reduce_accumulate [options]

Options:
    --variant {all|helper|fast|dispatch} ...   which paths to run
    --dim     {all|row|col|scalar} ...         reduce dimension(s)
    --accum   {all|fp32|bf16} ...              accumulation (DEST/SFPU) dtype for the precision tables
    --widths  N [N ...]                        tile counts to reduce (default 1 2 4 8 16 32)
    --trials  N                                timed passes; median +/- std
    --kernel-iters K                           in-kernel loop count (K large = steady-state)
    --report  PATH                             write the report to PATH
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import DIMS, DTYPES, VARIANTS

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_reduce_accumulate.py::{}"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.reduce_accumulate")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--dim", nargs="+", choices=("all",) + DIMS, default=["all"])
    parser.add_argument("--accum", nargs="+", choices=("all",) + DTYPES, default=["all"])
    parser.add_argument("--widths", nargs="+", type=_positive)
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=200)
    parser.add_argument("--report")
    args = parser.parse_args()

    def expand(sel, allowed):
        return list(allowed) if "all" in sel else list(dict.fromkeys(sel))

    env = dict(
        os.environ,
        RA_VARIANTS=",".join(expand(args.variant, VARIANTS)),
        RA_DIMS=",".join(expand(args.dim, DIMS)),
        RA_ACCUMS=",".join(expand(args.accum, DTYPES)),
        RA_TRIALS=str(args.trials),
        RA_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.widths:
        env["RA_WIDTHS"] = ",".join(str(n) for n in args.widths)
    if args.report:
        env["RA_REPORT"] = args.report

    node = _TEST_NODE.format("test_reduce_accumulate_device_perf")
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", node], env=env)


if __name__ == "__main__":
    sys.exit(main())
