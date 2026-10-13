# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure trid_double_issue on YOUR shape/cores/block/trid-depth.

    python -m ttnn.operations.examples.trid_double_issue [--shape H,W] [--cores N]
                                                          [--blocks 1,2,4,8,16]
                                                          [--trids 2,3,4]
                                                          [--cb-blocks N]
                                                          [--dtype bfloat16]
                                                          [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, in-process profiler, post-run reset) and prints the same ns/op +
GB/s table (block x trid depth, against the full-barrier baseline). Run from the
repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_trid_double_issue.py::test_trid_double_issue_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.trid_double_issue")
    ap.add_argument("--shape", default="512,512", help="H,W of the tiled tensor (tile-aligned). Default 512,512.")
    ap.add_argument("--cores", type=int, default=1, help="cores running the copy (each independent). Default 1.")
    ap.add_argument(
        "--blocks", default="1,2,4,8,16", help="comma list of pages-per-barrier to sweep. Default 1,2,4,8,16."
    )
    ap.add_argument("--trids", default="2,3,4", help="comma list of trid depths (blocks in flight). Default 2,3,4.")
    ap.add_argument(
        "--ahead",
        default="1,2,3,4",
        help="baseline strength: blocks issued per barrier. 1 = naive loop, >1 = strongest "
        "non-trid reader (must be <= cb-blocks). Default 1,2,3,4.",
    )
    ap.add_argument(
        "--cb-blocks",
        type=int,
        default=6,
        help="CB depth in blocks, held FIXED across the table (must be >= max trid depth). Default 6.",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat8_b", "bfloat16", "float32"],
        help="tile format = transfer size (~1088/2048/4096 B). Default bfloat16.",
    )
    ap.add_argument("--iters", type=int, default=1, help="in-kernel repeat of the page range. 1=latency. Default 1.")
    ap.add_argument("--trials", type=int, default=10, help="profiled launches per case (averaged). Default 10.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        TDI_SHAPE=args.shape,
        TDI_CORES=str(args.cores),
        TDI_BLOCKS=args.blocks,
        TDI_TRIDS=args.trids,
        TDI_AHEAD=args.ahead,
        TDI_CB_BLOCKS=str(args.cb_blocks),
        TDI_DTYPE=args.dtype,
        TDI_ITERS=str(args.iters),
        TDI_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[trid_double_issue] shape={args.shape} cores={args.cores} blocks={args.blocks} "
        f"trids={args.trids} ahead={args.ahead} cb_blocks={args.cb_blocks} dtype={args.dtype} iters={args.iters}"
    )
    print(f"[trid_double_issue] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
