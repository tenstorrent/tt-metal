# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure whole-tile vs row-0-only reads of a row-broadcast operand on your own params.

    python -m ttnn.operations.examples.partial_tile_read [options]

Runs the device-perf test through scripts/run_safe_pytest.sh (device lock, hang detection), so run
it from the tt-metal root. Without --cores / --widths / --rows the predefined sweep runs.
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import ABLATION, DEFAULT_BLOCK, VARIANTS

_TEST = "tests/ttnn/unit_tests/operations/examples/test_partial_tile_read.py::test_partial_tile_read_device_perf"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _cores(value):
    if value != "full":
        _positive(value)
    return value


def _width(value):
    parsed = _positive(value)
    if parsed % 32:
        raise argparse.ArgumentTypeError("W must be a multiple of 32")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.partial_tile_read")
    methods = VARIANTS + (ABLATION,)
    parser.add_argument(
        "--variant", nargs="+", choices=("all",) + methods, default=["all"], help="methods to compare (default all)"
    )
    parser.add_argument("--trials", type=_positive, default=1, help="measured launches per method; median")
    parser.add_argument(
        "--kernel-iters", type=_positive, default=1, help="vector streams per launch; 1=latency, larger=steady state"
    )
    parser.add_argument("--report", help="also write the Markdown report to this path")
    parser.add_argument("--cores", type=_cores, nargs="+", help="Tensix core counts, or 'full' for the whole grid")
    parser.add_argument("--widths", type=_width, nargs="+", help="vector lengths W in elements (multiples of 32)")
    parser.add_argument("--rows", type=_positive, nargs="+", help="tile-rows of resident input per Tensix core")
    parser.add_argument(
        "--block", type=_positive, default=DEFAULT_BLOCK, help=f"tiles per read barrier (default {DEFAULT_BLOCK})"
    )
    args = parser.parse_args()

    selected = methods if "all" in args.variant else tuple(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        PTR_VARIANTS=",".join(selected),
        PTR_TRIALS=str(args.trials),
        PTR_KERNEL_ITERS=str(args.kernel_iters),
        PTR_BLOCK=str(args.block),
    )
    for name, values in (("PTR_CORES", args.cores), ("PTR_WIDTHS", args.widths), ("PTR_ROWS", args.rows)):
        if values:
            env[name] = ",".join(str(v) for v in values)
    if args.report:
        env["PTR_REPORT"] = os.path.abspath(args.report)
    print(
        f"[partial_tile_read] variants={','.join(selected)} cores={args.cores or 'sweep'} "
        f"widths={args.widths or 'sweep'} rows={args.rows or 'sweep'} block={args.block} "
        f"kernel-iters={args.kernel_iters} trials={args.trials}"
    )
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
