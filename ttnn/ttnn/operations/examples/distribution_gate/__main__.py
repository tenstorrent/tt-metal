# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure distribution_gate on YOUR aspect-ratio sweep.

    python -m ttnn.operations.examples.distribution_gate
        [--shapes 32x4096,2048x32,2048x2048,1024x1024]
        [--variant all|height_split|width_split|gated]
        [--dtype bfloat16|float32|bfloat8_b] [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run reset),
printing the ns/op + active-cores table (height_split vs width_split vs gated,
per shape). Run from the repo root with the Python env active. Shapes are HxW;
sweep the aspect ratio to watch each fixed split collapse on its bad regime
while the gate fills the grid on both.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_distribution_gate.py::test_distribution_gate_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.distribution_gate")
    ap.add_argument(
        "--shapes",
        default="32x4096,2048x32,2048x2048,1024x1024",
        help="comma list of HxW shapes. Default 32x4096,2048x32,2048x2048,1024x1024.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="all | height_split | width_split | gated (which to run). Default all.",
    )
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
        DG_SHAPES=args.shapes,
        DG_VARIANT=args.variant,
        DG_DTYPE=args.dtype,
        DG_ITERS=str(args.iters),
        DG_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[distribution_gate] shapes={args.shapes} variant={args.variant} dtype={args.dtype} iters={args.iters}")
    print(f"[distribution_gate] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
