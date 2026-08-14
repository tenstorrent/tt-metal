# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure the kernel-fold program-structure lever (compute_only vs reader_compute_writer).

The lever is op-agnostic — does folding the reader/writer into the compute kernel help, or do the
separate dataflow kernels (running concurrently on NCRISC/BRISC) earn their keep? The payload is an
incidental same-spec zero-copy sharded tilize, used only to isolate program structure (no DRAM/NoC).

    python -m ttnn.operations.examples.zero_copy_fold [options]

Options:
    --variant {all|compute_only|reader_compute_writer} ...   which program structure(s) to compare
    --shape   H W NCORES                                     a case (repeatable); overrides the sweep
    --trials  N                                              timed passes; median +/- std
    --kernel-iters K                                         in-kernel loop count (K large = steady-state)
    --report  PATH                                           write the report to PATH
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import VARIANTS

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_zero_copy_fold.py::test_zero_copy_fold_device_perf"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.zero_copy_fold")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--shape", nargs=3, type=_positive, action="append", metavar=("H", "W", "NCORES"))
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=100)
    parser.add_argument("--report")
    args = parser.parse_args()

    variants = list(VARIANTS) if "all" in args.variant else list(dict.fromkeys(args.variant))
    env = dict(
        os.environ,
        ZCF_VARIANTS=",".join(variants),
        ZCF_TRIALS=str(args.trials),
        ZCF_KERNEL_ITERS=str(args.kernel_iters),
    )
    if args.shape:
        env["ZCF_SHAPES"] = ";".join(f"{h},{w},{n}" for h, w, n in args.shape)
    if args.report:
        env["ZCF_REPORT"] = args.report

    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST_NODE], env=env)


if __name__ == "__main__":
    sys.exit(main())
