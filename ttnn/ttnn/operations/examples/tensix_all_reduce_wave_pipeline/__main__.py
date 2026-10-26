# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import MAX_WAVES

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_wave_pipeline.py"
    "::test_tensix_all_reduce_wave_pipeline_device_perf"
)


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.tensix_all_reduce_wave_pipeline")
    parser.add_argument(
        "--waves",
        type=_positive,
        nargs="+",
        default=[1, 2, 4, 8],
        help=f"wave counts to compare against one_shot (max {MAX_WAVES}); counts the geometry cannot split "
        "evenly are skipped; default: 1 2 4 8",
    )
    parser.add_argument("--group-size", type=_positive, help="Tensix cores per group along a grid row; default: 8")
    parser.add_argument("--num-groups", type=_positive, help="groups, one per grid row; default: 8")
    parser.add_argument(
        "--rows", type=_positive, help="tile-rows per Tensix core; omit (with --width-tiles) for the default cases"
    )
    parser.add_argument("--width-tiles", type=_positive, default=4, help="tiles per tile-row per Tensix core")
    parser.add_argument("--trials", type=_positive, default=1, help="measured launches per method; default: 1")
    parser.add_argument(
        "--kernel-iters",
        type=_positive,
        default=1,
        help="all-reduces inside one kernel launch; 1=latency, larger=steady state",
    )
    parser.add_argument("--report", help="optional Markdown report path")
    args = parser.parse_args()

    env = dict(
        os.environ,
        WP_WAVES=",".join(str(w) for w in args.waves),
        WP_TRIALS=str(args.trials),
        WP_KERNEL_ITERS=str(args.kernel_iters),
    )
    for name, value in (("WP_GROUP_SIZE", args.group_size), ("WP_NUM_GROUPS", args.num_groups)):
        if value is not None:
            env[name] = str(value)
    if args.rows is not None:
        env["WP_ROWS"] = str(args.rows)
        env["WP_WIDTH_TILES"] = str(args.width_tiles)
    if args.report:
        env["WP_REPORT"] = args.report

    print(
        f"[tensix_all_reduce_wave_pipeline] rows={args.rows or 'default cases'} width_tiles={args.width_tiles} "
        f"waves={args.waves} kernel-iters={args.kernel_iters} trials={args.trials}"
    )
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
