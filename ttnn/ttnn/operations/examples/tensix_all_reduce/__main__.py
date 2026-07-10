# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import VARIANTS

_TEST = "tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py::test_tensix_all_reduce_device_perf"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.tensix_all_reduce")
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=("all",) + VARIANTS,
        default=["all"],
        help="algorithm(s) to measure; default: all",
    )
    parser.add_argument(
        "--group-shape",
        help="rectangular group as ROWS,COLS; omit to run grid-derived placements",
    )
    parser.add_argument("--num-groups", type=_positive, default=1, help="number of packed groups; default: 1")
    parser.add_argument("--num-tiles", type=_positive, default=6, help="tiles contributed by each core; default: 6")
    parser.add_argument(
        "--trials", type=_positive, default=5, help="profile trials after the discarded trial; default: 5"
    )
    parser.add_argument(
        "--kernel-iters",
        type=_positive,
        default=1,
        help="all-reduces inside one kernel launch; 1=latency, larger=steady state",
    )
    parser.add_argument("--report", help="optional Markdown report path")
    args = parser.parse_args()

    group_shape = None
    if args.group_shape is not None:
        try:
            group_shape = tuple(int(part) for part in args.group_shape.split(","))
        except ValueError:
            parser.error("--group-shape must be ROWS,COLS")
        if len(group_shape) != 2 or any(dimension < 1 for dimension in group_shape):
            parser.error("--group-shape must contain two positive dimensions")

    selected = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        AR_NUM_TILES=str(args.num_tiles),
        AR_VARIANTS=",".join(selected),
        AR_TRIALS=str(args.trials),
        AR_KERNEL_ITERS=str(args.kernel_iters),
    )
    if group_shape is not None:
        env["AR_GROUP_SHAPE"] = f"{group_shape[0]},{group_shape[1]}"
        env["AR_NUM_GROUPS"] = str(args.num_groups)
    else:
        env.pop("AR_GROUP_SHAPE", None)
        env.pop("AR_NUM_GROUPS", None)
    if args.report:
        env["AR_REPORT"] = args.report

    command = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[tensix_all_reduce] group={args.group_shape or 'grid-derived sweep'} "
        f"groups={args.num_groups if group_shape is not None else 'grid-derived'} tiles={args.num_tiles} "
        f"variants={','.join(selected)} kernel-iters={args.kernel_iters} trials={args.trials}"
    )
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
