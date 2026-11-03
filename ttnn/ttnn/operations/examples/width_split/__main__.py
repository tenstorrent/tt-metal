# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure width_split on YOUR width sweep.

    python -m ttnn.operations.examples.width_split [--widths 32,256,1024,2048,4096,8192]
                                                    [--variant all|single_core|width_split]
                                                    [--iters K] [--trials N]

    # reduce case (per-row mean subtracted from the row; a width split needs a cross-core combine)
    python -m ttnn.operations.examples.width_split --reduce [--rows 1,16,64,128] [--wts 4,8,16,32,64]
                                                    [--ks 1,2,4,8,16,32,64] [--max-oversub 2]
                                                    [--iters K] [--trials N]

    # reduce case on ROW_MAJOR sticks (one DRAM page = one row of W = 32 * Wt bf16 datums; a split
    # into k slices reads / writes k pieces of W * 2 / k B per stick instead of one whole stick)
    python -m ttnn.operations.examples.width_split --reduce --layout row_major
                                                    [--rows 1,4,16,47,64,128] [--wts 2,4,8,16,32,64]
                                                    [--ks 1,2,4] [--max-oversub 2] [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run reset),
printing the same ns/op + cores + speedup table (single_core vs width_split,
per width). Run from the repo root with the Python env active. H is fixed at 32
(one tile-row — the wide-short case); only the width sweeps.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_device_perf"
_REDUCE_TEST = "tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_device_perf"
_REDUCE_RM_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_rm_device_perf"
)
# --reduce defaults per --layout: (rows, wts, ks)
_REDUCE_DEFAULTS = {
    "tile": ("1,16,64,128", "4,8,16,32,64", "1,2,4,8,16,32,64"),
    "row_major": ("1,4,16,47,64,128", "2,4,8,16,32,64", "1,2,4"),
}


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.width_split")
    ap.add_argument(
        "--widths",
        default="32,256,1024,2048,4096,8192",
        help="comma list of tensor widths W (H fixed at 32). Default 32,256,1024,2048,4096,8192.",
    )
    ap.add_argument("--variant", default="all", help="all | single_core | width_split (which to run). Default all.")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat8_b", "bfloat16", "float32"],
        help="tile format. Default bfloat16.",
    )
    ap.add_argument("--iters", type=int, default=1, help="in-kernel repeat (K). 1=latency, large=steady. Default 1.")
    ap.add_argument("--trials", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    ap.add_argument(
        "--reduce",
        action="store_true",
        help="run the reduce case (row_split_reduce vs width_split_reduce) instead of the elementwise one.",
    )
    ap.add_argument(
        "--layout",
        default="tile",
        choices=["tile", "row_major"],
        help="reduce case: input/output Layout (tile = 2 KB tile pages; row_major = W * 2 B stick pages, "
        "tilized/untilized in the kernel). Default tile.",
    )
    ap.add_argument(
        "--rows",
        default=None,
        help="reduce case: tile-rows R_t to sweep. Default 1,16,64,128 (tile) / 1,4,16,47,64,128 (row_major).",
    )
    ap.add_argument(
        "--wts",
        default=None,
        help="reduce case: tile-columns Wt to sweep (W = 32 * Wt; a stick is 64 * Wt B). "
        "Default 4,8,16,32,64 (tile) / 2,4,8,16,32,64 (row_major).",
    )
    ap.add_argument(
        "--ks",
        default=None,
        help="reduce case: split factors k to try (kept when k divides Wt). "
        "Default 1,2,4,8,16,32,64 (tile) / 1,2,4 (row_major).",
    )
    ap.add_argument(
        "--max-oversub",
        type=float,
        default=2.0,
        help="reduce case: skip k with R_t * k > this * grid Tensix cores. Default 2.",
    )
    args = ap.parse_args()

    if args.reduce:
        d_rows, d_wts, d_ks = _REDUCE_DEFAULTS[args.layout]
        rows, wts, ks = args.rows or d_rows, args.wts or d_wts, args.ks or d_ks
        prefix = "WSRM" if args.layout == "row_major" else "WSR"
        env = dict(
            os.environ,
            WSR_MAX_OVERSUB=str(args.max_oversub),
            WS_ITERS=str(args.iters),
            WS_TRIALS=str(args.trials),
        )
        env.update({f"{prefix}_ROWS": rows, f"{prefix}_WTS": wts, f"{prefix}_KS": ks})
        test = _REDUCE_RM_TEST if args.layout == "row_major" else _REDUCE_TEST
        cmd = ["scripts/run_safe_pytest.sh", "--run-all", test]
        print(f"[width_split --reduce] layout={args.layout} rows={rows} wts={wts} ks={ks} iters={args.iters}")
        print(f"[width_split --reduce] {' '.join(cmd)}")
        return subprocess.call(cmd, env=env)

    env = dict(
        os.environ,
        WS_WIDTHS=args.widths,
        WS_VARIANT=args.variant,
        WS_DTYPE=args.dtype,
        WS_ITERS=str(args.iters),
        WS_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[width_split] widths={args.widths} variant={args.variant} dtype={args.dtype} iters={args.iters}")
    print(f"[width_split] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
