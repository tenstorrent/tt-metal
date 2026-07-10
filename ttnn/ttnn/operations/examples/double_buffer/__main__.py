# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure double_buffer on YOUR shape/cores/reads-per-barrier.

    python -m ttnn.operations.examples.double_buffer [--shape H,W] [--cores N]
                                                      [--blocks 1,2,4,8,16,32]
                                                      [--passes K] [--iters K]
                                                      [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, in-process profiler, post-run reset) and prints the same ns/op +
GB/s table (block x single/double buffered). Run from the repo root with the
Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.double_buffer")
    ap.add_argument("--shape", default="512,512", help="H,W of the bf16 tiled tensor (tile-aligned). Default 512,512.")
    ap.add_argument("--cores", type=int, default=1, help="cores running the pipeline (each independent). Default 1.")
    ap.add_argument(
        "--blocks",
        default="1,2,4,8,16,32",
        help="comma list of reads/writes-per-barrier to sweep. Default 1,2,4,8,16,32.",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat8_b", "bfloat16", "float32"],
        help="tile format = transfer size (~1088/2048/4096 B). Default bfloat16.",
    )
    ap.add_argument("--passes", type=int, default=1, help="compute_passes (relu repeats), kept light. Default 1.")
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="in-kernel repeat of the tile range (K). 1=latency, large=steady. Default 1.",
    )
    ap.add_argument("--trials", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        DB_SHAPE=args.shape,
        DB_CORES=str(args.cores),
        DB_BLOCKS=args.blocks,
        DB_DTYPE=args.dtype,
        DB_PASSES=str(args.passes),
        DB_ITERS=str(args.iters),
        DB_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[double_buffer] shape={args.shape} cores={args.cores} blocks={args.blocks} dtype={args.dtype} passes={args.passes} iters={args.iters}"
    )
    print(f"[double_buffer] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
