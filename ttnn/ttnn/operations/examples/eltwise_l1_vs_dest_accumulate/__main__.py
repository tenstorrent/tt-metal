# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure L1-accumulate vs read-modify-write on YOUR accumulation loop.

    python -m ttnn.operations.examples.eltwise_l1_vs_dest_accumulate [--blocks 64] [--iters 100] [--trials 10] [--report PATH]

Runs the device-perf test through scripts/run_safe_pytest.sh (device lock, in-process profiler,
post-run reset) and prints the rmw vs pack_l1_acc table. Run from the repo root with the env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.eltwise_l1_vs_dest_accumulate")
    ap.add_argument(
        "--blocks", type=int, default=64, help="accumulation steps (single-tile blocks summed). Default 64."
    )
    ap.add_argument("--iters", type=int, default=100, help="in-kernel repeat (steady-state). Default 100.")
    ap.add_argument("--trials", type=int, default=10, help="profiled rounds (median). Default 10.")
    ap.add_argument("--report", help="also write the markdown report to PATH (nothing is written by default)")
    args = ap.parse_args()

    env = dict(
        os.environ,
        ELDA_B=str(args.blocks),
        ELDA_ITERS=str(args.iters),
        ELDA_TRIALS=str(args.trials),
    )
    if args.report:
        env["ELDA_REPORT"] = args.report
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[eltwise_l1_vs_dest_accumulate] blocks={args.blocks} iters={args.iters}")
    print(f"[eltwise_l1_vs_dest_accumulate] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
