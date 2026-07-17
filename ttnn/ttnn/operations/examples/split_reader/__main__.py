# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the split-reader off/on device-performance comparison."""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_split_reader.py::test_split_reader_device_perf"


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.split_reader")
    parser.add_argument("--cores", type=int, default=8, help="source shards in the top row (default: 8)")
    parser.add_argument(
        "--tiles-per-core", type=int, default=8, help="L1 source tiles owned by each worker (default: 8)"
    )
    parser.add_argument(
        "--block-tiles", type=int, default=8, help="input tiles assigned per streaming block (default: 8)"
    )
    parser.add_argument(
        "--transaction-bytes",
        type=int,
        nargs="+",
        default=[64],
        help="one or more NoC read sizes to compare (default: 64)",
    )
    parser.add_argument("--iters", type=int, default=20, help="profiled launches per variant (default: 20)")
    args = parser.parse_args()

    env = dict(
        os.environ,
        SR_CORES=str(args.cores),
        SR_TILES_PER_CORE=str(args.tiles_per_core),
        SR_BLOCK_TILES=str(args.block_tiles),
        SR_TRANSACTION_BYTES_LIST=",".join(str(value) for value in args.transaction_bytes),
        SR_ITERS=str(args.iters),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[split_reader] cores={args.cores} tiles_per_core={args.tiles_per_core} block_tiles={args.block_tiles} "
        f"transaction_bytes={args.transaction_bytes} iters={args.iters}"
    )
    print(f"[split_reader] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
