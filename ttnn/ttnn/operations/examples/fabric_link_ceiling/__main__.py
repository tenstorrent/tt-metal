# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure fabric_link_ceiling on YOUR fabric / payload / link count / stream size.

    python -m ttnn.operations.examples.fabric_link_ceiling [--fabric 1d,2d] [--payload 4352,8704,14336,15232]
                                                          [--links 1,2] [--mb 64] [--variant all]
                                                          [--direction all] [--trials 3]

Translates the flags into env overrides and runs the device-perf test through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). Needs a 2x2 mesh (4 chips). Run from the repo root
with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_fabric_link_ceiling.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.fabric_link_ceiling")
    ap.add_argument("--fabric", default="1d,2d", help="comma list of fabric configs: 1d, 2d. Default 1d,2d.")
    ap.add_argument(
        "--payload",
        default="4352,8704,14336,15232",
        help="comma list of router max payloads in bytes (one packet = one payload). Default 4352,8704,14336,15232.",
    )
    ap.add_argument("--links", default="1,2", help="comma list of link counts (one sender core per link). Default 1,2.")
    ap.add_argument("--mb", type=float, default=64, help="MiB streamed per link per direction per launch. Default 64.")
    ap.add_argument("--variant", default="all", help="all | flush_per_packet | header_ring (comma list). Default all.")
    ap.add_argument("--direction", default="all", help="all | uni | bi (comma list). Default all.")
    ap.add_argument("--trials", type=int, default=3, help="measured launches per case (median). Default 3.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        FLC_FABRICS=args.fabric,
        FLC_PAYLOADS=args.payload,
        FLC_LINKS=args.links,
        FLC_MB=str(args.mb),
        FLC_TRIALS=str(args.trials),
    )
    if args.variant != "all":
        env["FLC_VARIANTS"] = args.variant
    if args.direction != "all":
        env["FLC_DIRECTIONS"] = args.direction
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-s"]
    print(f"[fabric_link_ceiling] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
