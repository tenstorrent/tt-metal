# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure fabric_all_gather on YOUR fabric configs / topologies / links / shard.

    python -m ttnn.operations.examples.fabric_all_gather [--fabric 1d,2d] [--topology 4x1_line,4x1_ring]
                                                        [--links 1,2] [--shape 2048,4096] [--dtype bf16]
                                                        [--dim 0] [--payload 14336] [--trials 3]

Translates the flags into env overrides and runs the device-perf test through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). Needs 4 chips. Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.fabric_all_gather")
    ap.add_argument(
        "--fabric", default=None, help="comma list: 1d, 1d_ring, 1d_neighbor_exchange, 2d, 2d_torus_x/y/xy."
    )
    ap.add_argument(
        "--topology", default=None, help="comma list, e.g. 2x2_axis0_line,2x2_snake_ring,4x1_line,4x1_ring."
    )
    ap.add_argument("--links", default="1,2", help="comma list of links per hop. Default 1,2.")
    ap.add_argument("--shape", default="2048,4096", help="per-chip shard H,W (tile-aligned). Default 2048,4096.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "bfp8", "fp32"], help="shard dtype. Default bf16.")
    ap.add_argument("--dim", default="0", choices=["0", "2"], help="gather dim: 0, or 2 (= dim -2). Default 0.")
    ap.add_argument("--payload", type=int, default=14336, help="router max payload in bytes. Default 14336.")
    ap.add_argument("--trials", type=int, default=3, help="measured launches per case (median). Default 3.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        AG_LINKS=args.links,
        AG_SHAPE=args.shape,
        AG_DTYPE=args.dtype,
        AG_DIM=args.dim,
        AG_PAYLOAD=str(args.payload),
        AG_TRIALS=str(args.trials),
    )
    if args.fabric:
        env["AG_FABRICS"] = args.fabric
    if args.topology:
        env["AG_TOPOS"] = args.topology
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-s"]
    print(f"[fabric_all_gather] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
