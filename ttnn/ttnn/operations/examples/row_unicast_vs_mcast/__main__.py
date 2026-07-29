# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import VARIANTS

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_row_unicast_vs_mcast.py::" "test_row_unicast_vs_mcast_device_perf"
)


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.row_unicast_vs_mcast")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--num-rows", type=_positive, help="active hardware rows; default: all")
    parser.add_argument(
        "--row-width",
        nargs="+",
        type=_positive,
        help="one or more core counts per row; default: 2, 4, 8, and full grid width",
    )
    parser.add_argument("--num-tiles", nargs="+", type=_positive, default=[1, 4, 16])
    parser.add_argument(
        "--num-writes",
        nargs="+",
        type=_positive,
        default=[1, 2, 4, 8, 16, 32],
        help="equal-sized NoC dispatches used to move each fixed-size payload",
    )
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=100)
    parser.add_argument("--report")
    args = parser.parse_args()

    selected = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        ROW_BCAST_VARIANTS=",".join(selected),
        ROW_BCAST_TILES=",".join(str(value) for value in args.num_tiles),
        ROW_BCAST_WRITES=",".join(str(value) for value in args.num_writes),
        ROW_BCAST_TRIALS=str(args.trials),
        ROW_BCAST_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.num_rows is not None:
        env["ROW_BCAST_ROWS"] = str(args.num_rows)
    else:
        env.pop("ROW_BCAST_ROWS", None)
    if args.row_width is not None:
        env["ROW_BCAST_WIDTHS"] = ",".join(str(value) for value in args.row_width)
    else:
        env.pop("ROW_BCAST_WIDTHS", None)
    if args.report:
        env["ROW_BCAST_REPORT"] = args.report

    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
