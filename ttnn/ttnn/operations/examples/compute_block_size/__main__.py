# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure compute block size for (A + B) @ C on your own shape.

    python -m ttnn.operations.examples.compute_block_size [options]

Options:
    --variant {all|per_tile_row|block2|block4|one_block} ...  which block granularities to run
    --m-tiles N            M height in tiles (rows / 32)          (default 8)
    --k-tiles N            K inner dim in tiles                    (default 4)
    --n-tiles N            N width in tiles                        (default 4)
    --trials N             timed passes; median +/- std
    --kernel-iters K       in-kernel loop count (K large = steady-state throughput)
    --report PATH          write the report table to PATH

A variant is skipped if its block height does not evenly divide --m-tiles
(per_tile_row=1, block2=2, block4=4, one_block=M_tiles).
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import VARIANTS

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_compute_block_size.py::{}"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.compute_block_size")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--m-tiles", type=_positive, default=8)
    parser.add_argument("--k-tiles", type=_positive, default=4)
    parser.add_argument("--n-tiles", type=_positive, default=4)
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=100)
    parser.add_argument("--report")
    args = parser.parse_args()

    variants = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))

    env = dict(
        os.environ,
        CBS_VARIANTS=",".join(variants),
        CBS_M_TILES=str(args.m_tiles),
        CBS_K_TILES=str(args.k_tiles),
        CBS_N_TILES=str(args.n_tiles),
        CBS_TRIALS=str(args.trials),
        CBS_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.report:
        env["CBS_REPORT"] = args.report

    node = _TEST_NODE.format("test_compute_block_size_device_perf")
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", node], env=env)


if __name__ == "__main__":
    sys.exit(main())
