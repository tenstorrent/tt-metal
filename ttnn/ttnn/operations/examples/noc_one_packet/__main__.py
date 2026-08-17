# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure noc_one_packet on YOUR page sizes / page counts / core count.

    python -m ttnn.operations.examples.noc_one_packet
        [--page-elems 32,128,512,1024,2048] [--pages-per-core 32] [--cores 8]
        [--kernel-iters 1] [--trials 3] [--variant all|generic|one_packet|generic_runtime_size]

Flags become env overrides for the device-perf test, which is then run through
scripts/run_safe_pytest.sh (device lock, profiler env, post-run reset). Run from the
repo root with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_noc_one_packet.py::test_noc_one_packet_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="noc_one_packet", description=__doc__)
    ap.add_argument("--page-elems", help="comma-separated bf16 elements per page (page_bytes = 2x)")
    ap.add_argument("--pages-per-core", help="comma-separated pages each core writes")
    ap.add_argument("--cores", help="number of cores in the ring")
    ap.add_argument("--kernel-iters", help="in-kernel repeats: 1 = per-launch latency, large = steady state")
    ap.add_argument("--trials", help="measured passes (median reported)")
    ap.add_argument("--variant", help=f"all | comma-separated subset")
    args = ap.parse_args()

    env = dict(os.environ)
    for flag, var in (
        ("page_elems", "N1P_PAGE_ELEMS"),
        ("pages_per_core", "N1P_PAGES"),
        ("cores", "N1P_CORES"),
        ("kernel_iters", "N1P_KERNEL_ITERS"),
        ("trials", "N1P_TRIALS"),
        ("variant", "N1P_VARIANT"),
    ):
        value = getattr(args, flag)
        if value:
            env[var] = value

    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST, "-q", "-s"]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
