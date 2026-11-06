# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import (
    ALL_ROW_VECTOR_VARIANTS,
    MEASURED_NULLS,
    RECOMMENDED,
    ROW_VECTOR_VARIANTS,
    VARIANTS,
)

_TEST_FILE = "tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py"
_TEST = f"{_TEST_FILE}::test_tensix_all_reduce_device_perf"
_ROW_VECTOR_TEST = f"{_TEST_FILE}::test_tensix_all_reduce_row_vector_sweep"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_list(value):
    return [_positive(part) for part in value.split(",") if part]


def _row_vector_main(args, parser):
    """Per-row-vector payload sweep: K members x b tile-rows x w tiles, every selected variant."""
    if "all" in args.variant:
        selected = list(ALL_ROW_VECTOR_VARIANTS)
    elif "recommended" in args.variant:
        selected = list(ROW_VECTOR_VARIANTS)
    else:
        selected = list(dict.fromkeys(args.variant))
        unknown = [v for v in selected if v not in ALL_ROW_VECTOR_VARIANTS]
        if unknown:
            parser.error(f"--row-vector variants must be among {', '.join(ALL_ROW_VECTOR_VARIANTS)}; got {unknown}")
    env = dict(
        os.environ,
        AR_RV_SWEEP="1",
        AR_RV_VARIANTS=",".join(selected),
        AR_RV_MEMBERS=",".join(str(k) for k in args.members),
        AR_RV_TILE_ROWS=",".join(str(b) for b in args.tile_rows),
        AR_RV_WIDTH_TILES=",".join(str(w) for w in args.width_tiles),
        AR_RV_TRIALS=str(args.trials),
        AR_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.report:
        env["AR_RV_REPORT"] = args.report
    print(
        f"[tensix_all_reduce --row-vector] members={args.members} tile-rows={args.tile_rows} "
        f"width-tiles={args.width_tiles} variants={','.join(selected)} kernel-iters={args.kernel_iters} "
        f"trials={args.trials}"
    )
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _ROW_VECTOR_TEST], env=env)


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.tensix_all_reduce")
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=("all", "recommended") + VARIANTS + tuple(v for v in ALL_ROW_VECTOR_VARIANTS if v not in VARIANTS),
        default=["recommended"],
        help=(
            "algorithm(s) to measure. Default 'recommended' skips the measured nulls "
            f"({', '.join(MEASURED_NULLS)}), which are ~4x off the best reducer at every size "
            "measured; 'all' includes them. With --row-vector: the row-vector variants "
            f"(recommended = {', '.join(ROW_VECTOR_VARIANTS)}; all = every partial x return x path)."
        ),
    )
    row = parser.add_argument_group(
        "per-row-vector payload (--row-vector): a row WIDTH_SHARDED over K Tensix cores, all-reduce of the row sums"
    )
    row.add_argument(
        "--row-vector",
        action="store_true",
        help="measure the per-row-vector all-reduce (partial size on the wire x return path) instead",
    )
    row.add_argument(
        "--members",
        type=_positive_list,
        default=[2, 4, 8, 12, 16, 32, 64, 128],
        help="K, Tensix cores the row is split across, comma-separated; default: 2,4,8,12,16,32,64,128",
    )
    row.add_argument(
        "--tile-rows",
        type=_positive_list,
        default=[1, 4],
        help="b, tile-rows per member (one owner per tile-row, so b <= K), comma-separated; default: 1,4",
    )
    row.add_argument(
        "--width-tiles",
        type=_positive_list,
        default=[1],
        help="w, tiles per member per tile-row, comma-separated; default: 1",
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
    if args.row_vector:
        return _row_vector_main(args, parser)
    bad = [v for v in args.variant if v not in ("all", "recommended") + VARIANTS]
    if bad:
        parser.error(f"{bad} are --row-vector variants")

    group_shape = None
    if args.group_shape is not None:
        try:
            group_shape = tuple(int(part) for part in args.group_shape.split(","))
        except ValueError:
            parser.error("--group-shape must be ROWS,COLS")
        if len(group_shape) != 2 or any(dimension < 1 for dimension in group_shape):
            parser.error("--group-shape must contain two positive dimensions")

    if "all" in args.variant:
        selected = list(VARIANTS)
    elif "recommended" in args.variant:
        selected = list(RECOMMENDED)
    else:
        selected = list(dict.fromkeys(args.variant))
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
