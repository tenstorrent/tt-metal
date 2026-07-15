# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure per-core DRAM read vs read-once+multicast of a shared input stream on your own shape.

    python -m ttnn.operations.examples.shared_input_reuse [--chunk-rows 16] [--d-cols 4] [--chunks 19] [--trials 10]

Runs the device-perf test through scripts/run_safe_pytest.sh (device lock, in-process profiler,
post-run reset) and prints the per_core_dram vs mcast table. Run from the repo root with the env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.shared_input_reuse")
    ap.add_argument("--chunk-rows", type=int, default=16, help="tile-rows per chunk. Default 16.")
    ap.add_argument("--d-cols", type=int, default=4, help="input width in tiles (tile-cols). Default 4.")
    ap.add_argument("--chunks", type=int, default=19, help="number of chunks. Default 19.")
    ap.add_argument("--trials", type=int, default=10, help="profiled rounds (median). Default 10.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        SIR_CHUNK_ROWS=str(args.chunk_rows),
        SIR_D_COLS=str(args.d_cols),
        SIR_CHUNKS=str(args.chunks),
        SIR_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[shared_input_reuse] chunk_rows={args.chunk_rows} d_cols={args.d_cols} chunks={args.chunks}")
    print(f"[shared_input_reuse] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
