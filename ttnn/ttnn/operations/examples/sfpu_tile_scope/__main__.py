# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure the isolated MATH-thread cost of a whole-tile vs scoped SFPU op.

    python -m ttnn.operations.examples.sfpu_tile_scope [options]

Options:
    --variant {all|none|rc|r|c|r_iter2|c_skip|face|face_iter1} ...  which scope(s) to compare (default all)
    --func    {all|rsqrt|recip} ...                  which SFPU op(s) (default all)
    --reps    N                                      in-kernel math-loop trip count (amortizes the
                                                     one zone marker; larger = cleaner) (default 2000)
    --trials  N                                      measured launches; median +/- std
    --report  PATH                                   write the report to PATH
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import FUNCS, VARIANTS

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_sfpu_tile_scope.py::{}"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.sfpu_tile_scope")
    parser.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    parser.add_argument("--func", nargs="+", choices=("all",) + FUNCS, default=["all"])
    parser.add_argument("--reps", type=_positive, default=2000)
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--report")
    args = parser.parse_args()

    def expand(sel, allowed):
        return list(allowed) if "all" in sel else list(dict.fromkeys(sel))

    env = dict(
        os.environ,
        STS_VARIANTS=",".join(expand(args.variant, VARIANTS)),
        STS_FUNCS=",".join(expand(args.func, FUNCS)),
        STS_REPS=str(args.reps),
        STS_TRIALS=str(args.trials),
    )
    if args.report:
        env["STS_REPORT"] = args.report

    node = _TEST_NODE.format("test_sfpu_tile_scope_device_perf")
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", node], env=env)


if __name__ == "__main__":
    sys.exit(main())
