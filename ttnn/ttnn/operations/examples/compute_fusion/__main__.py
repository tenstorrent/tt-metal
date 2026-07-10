# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure compute fusion vs L1 round-trips on your own shapes.

    python -m ttnn.operations.examples.compute_fusion [options]

Options:
    --scenario {all|sfpu_chain|fpu_sfpu|reduce_recip} ...  which expression(s) to run
    --tiles N ...          tile counts to sweep (reduce_recip: reduce width in tiles)
    --blocks N ...         DEST-lane block sizes (eltwise scenarios only; init hoisting)
    --trials N             timed passes; median +/- std
    --kernel-iters K       in-kernel loop count (K large = steady-state throughput)
    --report PATH          write the report table to PATH
    --microbench           per-phase DeviceZoneScopedN breakdown (unpack/math/pack ns per phase)
                           instead of the whole-kernel A/B sweep; uses --tiles[0] as the size
"""

import argparse
import os
import subprocess
import sys

from .program_descriptor_with_inline_kernels import SCENARIOS

_TEST_NODE = "tests/ttnn/unit_tests/operations/examples/test_compute_fusion.py::{}"


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.compute_fusion")
    parser.add_argument("--scenario", nargs="+", choices=("all",) + SCENARIOS, default=["all"])
    parser.add_argument("--tiles", nargs="+", type=_positive, default=[4, 16, 64])
    parser.add_argument("--blocks", nargs="+", type=_positive, default=[1, 4])
    parser.add_argument("--trials", type=_positive, default=5)
    parser.add_argument("--kernel-iters", type=_positive, default=100)
    parser.add_argument("--report")
    parser.add_argument("--microbench", action="store_true")
    args = parser.parse_args()

    scenarios = list(SCENARIOS) if "all" in args.scenario else list(dict.fromkeys(args.scenario))
    env = dict(os.environ, CF_SCENARIOS=",".join(scenarios))
    if args.microbench:
        env["CF_MB_TILES"] = str(args.tiles[0])
        env["CF_MB_KERNEL_ITERS"] = str(min(args.kernel_iters, 16))
        if args.report:
            env["CF_MB_REPORT"] = args.report
        node = _TEST_NODE.format("test_compute_fusion_microbench")
    else:
        env.update(
            CF_TILES=",".join(str(v) for v in args.tiles),
            CF_BLOCKS=",".join(str(v) for v in args.blocks),
            CF_TRIALS=str(args.trials),
            CF_KERNEL_ITERS=str(args.kernel_iters),
        )
        if args.report:
            env["CF_REPORT"] = args.report
        node = _TEST_NODE.format("test_compute_fusion_device_perf")
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", node], env=env)


if __name__ == "__main__":
    sys.exit(main())
