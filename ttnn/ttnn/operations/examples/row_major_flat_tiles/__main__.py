# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure tilize_untilize vs flat_tiles on your own shapes.

    python -m ttnn.operations.examples.row_major_flat_tiles \\
        [--case eltwise|reduce] [--op relu|gelu] [--reps K] [--rows H,...] [--widths W,...]
        [--reduce-algo auto|accumulate|reduce_tile] [--piece-order interleaved|row]
        [--variant all|tilize_untilize|flat_tiles ...] [--trials N] [--kernel-iters K] [--report PATH]

Without --case it runs the predefined sweep (eltwise relu / gelu / relu x16, reduce, plus the reduce
ablation); --rows / --widths narrow it. The flags become env overrides for the device-perf test (a
measurement sweep, opt-in: the CLI sets RMFT_SWEEPS=1), which runs through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). The report is printed; it is written to a file only
with --report PATH. Run from the repo root with python_env active.
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import CASES, ELTWISE_OPS, PIECE_ORDERS, REDUCE_ALGOS, VARIANTS

_TEST = "tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py::test_row_major_flat_tiles_device_perf"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _int_list(value):
    try:
        parsed = [int(v) for v in value.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    if any(v < 32 or v % 32 for v in parsed):
        raise argparse.ArgumentTypeError("every value must be a positive multiple of 32")
    return value


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.row_major_flat_tiles")
    ap.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"], help="default: all")
    ap.add_argument("--trials", type=_positive, default=1, help="measured passes (median); default 1")
    ap.add_argument(
        "--kernel-iters", type=_positive, default=1, help="in-kernel repeats; 1 = per-launch latency, large = steady"
    )
    ap.add_argument("--report", help="also write the Markdown report here (default: print only)")
    ap.add_argument("--case", choices=CASES, help="one case instead of the predefined sweep")
    ap.add_argument("--op", choices=ELTWISE_OPS, default="relu", help="eltwise op; default relu")
    ap.add_argument("--reps", type=_positive, default=1, help="eltwise: op passes in DEST per tile; default 1")
    ap.add_argument("--rows", type=_int_list, help="comma-separated H values (sticks), multiples of 32")
    ap.add_argument("--widths", type=_int_list, help="comma-separated W values (elements), multiples of 32")
    ap.add_argument("--reduce-algo", choices=REDUCE_ALGOS, default="auto", help="reduce datapath; default auto")
    ap.add_argument(
        "--piece-order", choices=PIECE_ORDERS, default=PIECE_ORDERS[0], help="reduce flat_tiles read / write order"
    )
    args = ap.parse_args()

    variants = VARIANTS if "all" in args.variant else tuple(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        RMFT_VARIANTS=",".join(variants),
        RMFT_TRIALS=str(args.trials),
        RMFT_KERNEL_ITERS=str(args.kernel_iters),
        RMFT_SWEEPS="1",  # the device-perf test is a measurement sweep, skipped unless enabled
    )
    for key in (
        "RMFT_CASE",
        "RMFT_OP",
        "RMFT_REPS",
        "RMFT_ROWS",
        "RMFT_WIDTHS",
        "RMFT_REDUCE_ALGO",
        "RMFT_PIECE_ORDER",
    ):
        env.pop(key, None)
    if args.case:
        env.update(
            RMFT_CASE=args.case,
            RMFT_OP=args.op,
            RMFT_REPS=str(args.reps),
            RMFT_REDUCE_ALGO=args.reduce_algo,
            RMFT_PIECE_ORDER=args.piece_order,
        )
    if args.rows:
        env["RMFT_ROWS"] = args.rows
    if args.widths:
        env["RMFT_WIDTHS"] = args.widths
    env["RMFT_REPORT"] = args.report or ""  # empty: print only
    what = f"case={args.case}" if args.case else "predefined sweep"
    print(
        f"[row_major_flat_tiles] {what} rows={args.rows or 'default'} widths={args.widths or 'default'} "
        f"variants={','.join(variants)} trials={args.trials} kernel-iters={args.kernel_iters}"
    )
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
