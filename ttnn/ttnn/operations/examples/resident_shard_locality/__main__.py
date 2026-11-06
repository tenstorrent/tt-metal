# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure resplit vs local vs local_alias on your own shard spec.

    python -m ttnn.operations.examples.resident_shard_locality \\
        [--layout height|block] [--cores grid|N|ROWS,COLS] [--shard-rows S --shard-width W] [--dtype bf16|fp32]
        [--suite all|base|wide]
        [--variant all|resplit|local|local_alias ...] [--trials N] [--kernel-iters K]
        [--swap-nocs] [--report PATH]

Without --shard-rows/--shard-width it runs the predefined sweep (--suite: all, the original
cases only, or only the wide-row / partial-width cases). The flags become env
overrides for the device-perf test, which runs through scripts/run_safe_pytest.sh (device
lock, in-process profiler, post-run reset). Run from the repo root with python_env active.
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import LAYOUTS, VARIANTS

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py"
    "::test_resident_shard_locality_device_perf"
)


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.resident_shard_locality")
    ap.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"], help="default: all")
    ap.add_argument("--trials", type=_positive, default=1, help="measured passes (median); default 1")
    ap.add_argument(
        "--kernel-iters", type=_positive, default=1, help="in-kernel repeats; 1 = per-launch latency, large = steady"
    )
    ap.add_argument("--report", help="also write the Markdown report here")
    ap.add_argument(
        "--layout", choices=LAYOUTS, default="height", help="HEIGHT_SHARDED or BLOCK_SHARDED; default height"
    )
    ap.add_argument(
        "--cores",
        default="grid",
        help="'grid' (whole compute grid), N Tensix cores row-major (height), or ROWS,COLS (block); default grid",
    )
    ap.add_argument("--shard-rows", type=_positive, help="sticks per Tensix core (shard height)")
    ap.add_argument("--shard-width", type=_positive, help="elements per stick (shard width)")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="element type; default bf16")
    ap.add_argument(
        "--suite", choices=("all", "base", "wide"), default="all", help="predefined sweep subset; default all"
    )
    ap.add_argument("--swap-nocs", action="store_true", help="ablation: reader on NoC1, writer on NoC0")
    args = ap.parse_args()

    if (args.shard_rows is None) != (args.shard_width is None):
        ap.error("--shard-rows and --shard-width go together")
    variants = VARIANTS if "all" in args.variant else tuple(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        RSL_VARIANTS=",".join(variants),
        RSL_TRIALS=str(args.trials),
        RSL_KERNEL_ITERS=str(args.kernel_iters),
        RSL_SWAP_NOCS="1" if args.swap_nocs else "0",
        RSL_SUITE=args.suite,
    )
    for key in ("RSL_LAYOUT", "RSL_CORES", "RSL_SHARD_ROWS", "RSL_SHARD_WIDTH", "RSL_DTYPE", "RSL_REPORT"):
        env.pop(key, None)
    if args.shard_rows is not None:
        env.update(
            RSL_LAYOUT=args.layout,
            RSL_CORES=args.cores,
            RSL_SHARD_ROWS=str(args.shard_rows),
            RSL_SHARD_WIDTH=str(args.shard_width),
            RSL_DTYPE=args.dtype,
        )
    env["RSL_REPORT"] = args.report or ""  # empty: print only, never overwrite the committed report.md
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    shard = (
        f"[{args.shard_rows}, {args.shard_width}] {args.dtype} {args.layout} cores={args.cores}"
        if args.shard_rows
        else f"sweep ({args.suite})"
    )
    print(
        f"[resident_shard_locality] {shard} variants={','.join(variants)} trials={args.trials} "
        f"kernel-iters={args.kernel_iters} swap-nocs={args.swap_nocs}"
    )
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
