# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure dram_saturation on YOUR shape / core sweep.

    python -m ttnn.operations.examples.dram_saturation
        [--shape 2048x2048] [--cores 1,2,4,8,16,32,48,64]
        [--variant all|spread|stacked] [--dtype bfloat16|float32|bfloat8_b]
        [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run reset),
printing the achieved-GB/s-vs-core-count table (spread vs stacked). Run from the
repo root with the Python env active. Sweep --cores to see bandwidth saturate:
the plateau onset is the sweet-spot core count; `stacked` saturates lower and can
roll over (more cores = slower).
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_dram_saturation.py::test_dram_saturation_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.dram_saturation")
    ap.add_argument("--shape", default="2048x2048", help="HxW of the copied tensor. Default 2048x2048 (DRAM-bound).")
    ap.add_argument("--cores", default="1,2,4,8,16,32,48,64", help="comma list of core counts to sweep. Default 1..64.")
    ap.add_argument("--variant", default="all", help="all | spread | stacked (which placement to run). Default all.")
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
        DS_SHAPE=args.shape,
        DS_CORES=args.cores,
        DS_VARIANT=args.variant,
        DS_DTYPE=args.dtype,
        DS_ITERS=str(args.iters),
        DS_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[dram_saturation] shape={args.shape} cores={args.cores} variant={args.variant} dtype={args.dtype}")
    print(f"[dram_saturation] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
