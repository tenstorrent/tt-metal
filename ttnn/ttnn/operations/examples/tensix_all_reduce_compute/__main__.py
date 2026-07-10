# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import VARIANTS

_TEST = (
    "tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_compute.py::"
    "test_tensix_all_reduce_compute_device_perf"
)


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.tensix_all_reduce_compute")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--num-blocks", nargs="+", type=_positive, default=[2, 4, 8, 16])
    parser.add_argument("--num-tiles", type=_positive, default=6)
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=100)
    parser.add_argument("--report")
    args = parser.parse_args()

    if any(num_blocks < 2 for num_blocks in args.num_blocks):
        parser.error("--num-blocks values must be at least 2")
    selected = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        ARC_COMPUTE_VARIANTS=",".join(selected),
        ARC_COMPUTE_BLOCKS=",".join(str(value) for value in args.num_blocks),
        ARC_COMPUTE_TILES=str(args.num_tiles),
        ARC_COMPUTE_TRIALS=str(args.trials),
        ARC_COMPUTE_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.report:
        env["ARC_COMPUTE_REPORT"] = args.report
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
