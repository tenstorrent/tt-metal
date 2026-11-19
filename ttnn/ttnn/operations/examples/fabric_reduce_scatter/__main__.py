# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure fabric_reduce_scatter on YOUR fabric configs / topologies / links / sizes.

    python -m ttnn.operations.examples.fabric_reduce_scatter [--fabric 2d_torus_xy] [--topology 4x1_line,4x1_ring]
                                                            [--links 1,2] [--workers 1,2] [--finals 2]
                                                            [--balance 1] [--shape 2048,4096] [--inc-every 8]
                                                            [--cb-kb 112] [--payload 14336] [--trials 5]

Translates the flags into env overrides and runs the device-perf test through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). Needs 4 chips. Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_fabric_reduce_scatter.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.fabric_reduce_scatter")
    ap.add_argument(
        "--fabric",
        default="2d_torus_xy",
        help="comma list: 1d, 1d_ring, 1d_neighbor_exchange, 2d, 2d_torus_x/y/xy. Default 2d_torus_xy.",
    )
    ap.add_argument(
        "--topology", default=None, help="comma list, e.g. 2x2_axis0_line,2x2_snake_ring,4x1_line,4x1_ring."
    )
    ap.add_argument("--links", default="1,2", help="comma list of links per hop. Default 1,2.")
    ap.add_argument("--workers", default="1", help="comma list: cores per link and direction (>1 = mux). Default 1.")
    ap.add_argument("--finals", default="2", help="comma list: final cores per link (with 1 worker). Default 2.")
    ap.add_argument("--balance", default="1", choices=["0", "1"], help="balanced ring (1) or not (0). Default 1.")
    ap.add_argument(
        "--shape", default="2048,4096", help="per-chip input H,W (tile-aligned), or H,W;H,W;... Default 2048,4096."
    )
    ap.add_argument("--inc-every", default="8", help="comma list: arrival increment every N chunks. Default 8.")
    ap.add_argument("--cb-kb", default="112", help="comma list: circular buffer per core, KiB. Default 112.")
    ap.add_argument("--payload", type=int, default=14336, help="router max payload in bytes. Default 14336.")
    ap.add_argument("--trials", type=int, default=5, help="measured calls per case (median). Default 5.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        RS_FABRICS=args.fabric,
        RS_LINKS=args.links,
        RS_WORKERS=args.workers,
        RS_FINALS=args.finals,
        RS_BALANCE=args.balance,
        RS_SHAPE=args.shape,
        RS_INC_EVERY=args.inc_every,
        RS_CB_KB=args.cb_kb,
        RS_PAYLOAD=str(args.payload),
        RS_TRIALS=str(args.trials),
        RS_BASELINE="0",
    )
    if args.topology:
        env["RS_TOPOS"] = args.topology
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-s"]
    print(f"[fabric_reduce_scatter] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
