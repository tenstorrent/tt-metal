# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure handshake_elision on YOUR shard geometry / core count.

    python -m ttnn.operations.examples.handshake_elision [--shards 1x2,1x4,2x4,4x4,4x8,8x8]
                                                          [--cores N] [--variant all|handshake,no_handshake]
                                                          [--iters K] [--trials N]

Translates the flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh, so measurement goes through the same proven path
(device lock, in-process profiler, post-run reset) and prints the same table.
Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

from .handshake_elision import VARIANTS

_TEST = "tests/ttnn/unit_tests/operations/examples/test_handshake_elision.py::test_handshake_elision_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.handshake_elision")
    ap.add_argument(
        "--shards",
        default="1x2,1x4,2x4,4x4,4x8,8x8",
        help="comma list of per-core shard geometries HTxWT in tiles (tiles/core = HT*WT). "
        "Default 1x2,1x4,2x4,4x4,4x8,8x8 (2..64 tiles/core).",
    )
    ap.add_argument("--cores", type=int, default=4, help="cores in row 0, one shard each. Default 4.")
    ap.add_argument("--variant", default="all", help="all, or a comma list from " + ",".join(VARIANTS))
    ap.add_argument("--iters", type=int, default=1, help="in-kernel repeat. 1 = per-launch latency. Default 1.")
    ap.add_argument("--trials", type=int, default=10, help="profiled launches per cell (averaged). Default 10.")
    args = ap.parse_args()

    variants = ",".join(VARIANTS) if args.variant == "all" else args.variant
    for v in variants.split(","):
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}; choose from {list(VARIANTS)}")

    env = dict(
        os.environ,
        HE_SHARDS=args.shards,
        HE_CORES=str(args.cores),
        HE_VARIANTS=variants,
        HE_ITERS=str(args.iters),
        HE_TRIALS=str(args.trials),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[handshake_elision] shards={args.shards} cores={args.cores} variants={variants} "
        f"iters={args.iters} trials={args.trials}"
    )
    print(f"[handshake_elision] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
