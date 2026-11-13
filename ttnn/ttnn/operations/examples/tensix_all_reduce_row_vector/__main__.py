# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import ALL_ROW_VECTOR_VARIANTS, ROW_VECTOR_VARIANTS

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_row_vector.py"
    "::test_tensix_all_reduce_row_vector_sweep"
)


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_list(value):
    return [_positive(part) for part in value.split(",") if part]


def main():
    """Per-row-vector payload sweep: K members x b tile-rows x w tiles, every selected variant."""
    parser = argparse.ArgumentParser(
        prog="python -m ttnn.operations.examples.tensix_all_reduce_row_vector",
        description="Per-row-vector all-reduce: a row WIDTH_SHARDED over K Tensix cores, all-reduce of the row sums.",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=("all", "recommended") + ALL_ROW_VECTOR_VARIANTS,
        default=["recommended"],
        help=(
            f"algorithm(s) to measure. Default 'recommended' = the measured set ({', '.join(ROW_VECTOR_VARIANTS)}); "
            "'all' = every partial x return x path combination."
        ),
    )
    parser.add_argument(
        "--members",
        type=_positive_list,
        default=[2, 4, 8, 12, 16, 32, 64, 128],
        help="K, Tensix cores the row is split across, comma-separated; default: 2,4,8,12,16,32,64,128",
    )
    parser.add_argument(
        "--tile-rows",
        type=_positive_list,
        default=[1, 4],
        help="b, tile-rows per member (one owner per tile-row, so b <= K), comma-separated; default: 1,4",
    )
    parser.add_argument(
        "--width-tiles",
        type=_positive_list,
        default=[1],
        help="w, tiles per member per tile-row, comma-separated; default: 1",
    )
    parser.add_argument(
        "--trials", type=_positive, default=5, help="interleaved profile trials per variant; default: 5"
    )
    parser.add_argument(
        "--kernel-iters",
        type=_positive,
        default=1,
        help="all-reduces inside one kernel launch; 1=latency, larger=steady state",
    )
    parser.add_argument("--report", help="optional Markdown report path (the sweep's tables are appended to it)")
    args = parser.parse_args()

    if "all" in args.variant:
        selected = list(ALL_ROW_VECTOR_VARIANTS)
    elif "recommended" in args.variant:
        selected = list(ROW_VECTOR_VARIANTS)
    else:
        selected = list(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        AR_RV_SWEEP="1",  # the measured test is a sweep: skipped unless this is set
        AR_RV_VARIANTS=",".join(selected),
        AR_RV_MEMBERS=",".join(str(k) for k in args.members),
        AR_RV_TILE_ROWS=",".join(str(b) for b in args.tile_rows),
        AR_RV_WIDTH_TILES=",".join(str(w) for w in args.width_tiles),
        AR_RV_TRIALS=str(args.trials),
        AR_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.report:
        env["AR_RV_REPORT"] = args.report
    else:
        env.pop("AR_RV_REPORT", None)
    print(
        f"[tensix_all_reduce_row_vector] members={args.members} tile-rows={args.tile_rows} "
        f"width-tiles={args.width_tiles} variants={','.join(selected)} kernel-iters={args.kernel_iters} "
        f"trials={args.trials}"
    )
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
