# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure the fused-CCL stats exchange on a 4-chip ring for YOUR slot sizes / links / core count.

    python -m ttnn.operations.examples.fabric_ring_exchange [--variant empty,fence,fence_direct,relay,direct]
                                                           [--bytes 128,2560] [--links 1,2] [--idle 0,100]
                                                           [--untraced 7] [--traced 8]

Translates the flags into env overrides and runs the device-perf test through scripts/run_safe_pytest.sh
(device lock, in-process profiler, post-run reset). Needs a 4-chip ring (1x4 mesh under FABRIC_2D_TORUS_XY).
Run from the repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_fabric_ring_exchange.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.fabric_ring_exchange")
    ap.add_argument("--variant", default="empty,fence,fence_direct,relay,direct", help="comma list of variants")
    ap.add_argument("--bytes", default="128,2560,8192,32768", help="comma list: slot bytes per link and wave")
    ap.add_argument("--links", default="1,2", help="comma list of links")
    ap.add_argument("--idle", default="100", help="comma list: idle compute cores in every program")
    ap.add_argument("--untraced", type=int, default=7, help="untraced calls per case (median)")
    ap.add_argument("--traced", type=int, default=8, help="traced back-to-back calls per case (median)")
    a = ap.parse_args()
    env = dict(
        os.environ,
        RX_VARIANTS=a.variant,
        RX_BYTES=a.bytes,
        RX_LINKS=a.links,
        RX_IDLE=a.idle,
        RX_UNTRACED=str(a.untraced),
        RX_TRACED=str(a.traced),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-s"]
    print(f"[fabric_ring_exchange] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
