# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure the row-mean accumulation methods on your own width / precision.

    python -m ttnn.operations.examples.row_reduce_accumulate [options]

Options:
    --variant      {all|reduce_fold|l1_accum|dest_accum|dest_accum_pairs} ...  which methods to run
    --precision    {all|fp32-fp32|bf16-fp32|bf16-bf16} ...   <input>-<accum> precision configs
    --distribution {all|signal|uniform|positive} ...   input value distribution(s) for accuracy
    --widths       E [E ...]  accumulation widths in ELEMENTS (multiples of 32; default 32..1024)
    --trials       N          timed passes; median +/- std
    --kernel-iters K          in-kernel loop count (K large = steady-state throughput)
    --report       PATH       write the report (overview + perf + accuracy tables) to PATH

Precision is "<input>-<accum>": fp32-fp32 (both precise), bf16-fp32 (lossy input, precise
accumulation), bf16-bf16 (+ accumulation loss). fp32-bf16 is intentionally not offered.
Distribution affects accuracy only (perf is data-independent): signal (per-row linspace+noise),
uniform [-1,1), positive [0,1).
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import METHODS, PRECISIONS

_DISTRIBUTIONS = ("signal", "uniform", "positive")

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_row_reduce_accumulate.py::{}"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _width_elems(value):
    parsed = int(value)
    if parsed < 1 or parsed % 32:
        raise argparse.ArgumentTypeError("width (elements) must be a positive multiple of 32")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.row_reduce_accumulate")
    parser.add_argument("--variant", nargs="+", choices=("all",) + METHODS, default=["all"])
    parser.add_argument("--precision", nargs="+", choices=("all",) + PRECISIONS, default=["all"])
    parser.add_argument("--distribution", nargs="+", choices=("all",) + _DISTRIBUTIONS, default=["all"])
    parser.add_argument("--widths", nargs="+", type=_width_elems)
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=200)
    parser.add_argument("--report")
    args = parser.parse_args()

    methods = list(METHODS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    precisions = list(PRECISIONS) if "all" in args.precision else list(dict.fromkeys(args.precision))
    dists = list(_DISTRIBUTIONS) if "all" in args.distribution else list(dict.fromkeys(args.distribution))

    env = dict(
        os.environ,
        RRA_METHODS=",".join(methods),
        RRA_PRECISIONS=",".join(precisions),
        RRA_DISTS=",".join(dists),
        RRA_TRIALS=str(args.trials),
        RRA_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.widths:
        env["RRA_WIDTHS"] = ",".join(str(w) for w in args.widths)
    if args.report:
        env["RRA_REPORT"] = args.report

    node = _TEST_NODE.format("test_row_reduce_accumulate_device_perf")
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", node], env=env)


if __name__ == "__main__":
    sys.exit(main())
