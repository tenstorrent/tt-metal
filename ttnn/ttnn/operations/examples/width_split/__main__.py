# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure width_split on YOUR width sweep.

    python -m ttnn.operations.examples.width_split [--widths 32,256,1024,2048,4096,8192]
                                                    [--variant all|single_core|width_split]
                                                    [--iters K] [--trials N]

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
    args = ap.parse_args()

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
