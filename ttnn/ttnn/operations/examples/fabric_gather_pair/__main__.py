# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure fabric_gather_pair on YOUR shard shape / placement / payload.

    python -m ttnn.operations.examples.fabric_gather_pair [--shape 8192,4096] [--variant all] [--payload 14336]
                                                         [--placements "a=2,0|b=2,0;3,0"] [--local-noc same,noc0]
                                                         [--ablate "|local_copy|fabric"] [--trials 3]

Translates the flags into env overrides and runs the device-perf test through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). Needs a 2x2 mesh (4 chips). Run from the repo root with the
Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_fabric_gather_pair.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.fabric_gather_pair")
    ap.add_argument("--shape", default="8192,4096", help="per-chip shard H,W (bf16, tile-aligned). Default 8192,4096.")
    ap.add_argument("--variant", default="all", help="all | page_per_packet | bank_run (comma list). Default all.")
    ap.add_argument("--payload", type=int, default=14336, help="router max payload in bytes. Default 14336.")
    ap.add_argument("--placements", default=None, help='link cores, e.g. "one=2,0|two=2,0;3,0" (logical x,y).')
    ap.add_argument("--local-noc", default="same,noc0", help="NoC for the local copy: same, noc0. Default both.")
    ap.add_argument("--ablate", default="", help='diagnostic ablation sets, e.g. "|local_copy|fabric+local_copy".')
    ap.add_argument("--trials", type=int, default=3, help="measured launches per case (median). Default 3.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        FGP_SHAPE=args.shape,
        FGP_PAYLOAD=str(args.payload),
        FGP_LOCAL_NOCS=args.local_noc,
        FGP_ABLATE=args.ablate,
        FGP_TRIALS=str(args.trials),
    )
    if args.variant != "all":
        env["FGP_VARIANTS"] = args.variant
    if args.placements:
        env["FGP_PLACEMENTS"] = args.placements
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-s"]
    print(f"[fabric_gather_pair] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
