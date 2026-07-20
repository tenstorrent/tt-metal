# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure noc_placement on YOUR shapes/params, or regenerate the report.

    # measure the read/write × NoC × placement matrix (device kernel ns/op)
    python -m ttnn.operations.examples.noc_placement [--shape H,W] [--cores N] [--block N]
                                                      [--kernel-iters N] [--iters N]

    # regenerate noc_placement_matrix.html from code + tt-npe (needs tt-npe built)
    python -m ttnn.operations.examples.noc_placement --report

The measure path translates flags into env overrides and runs the device-perf test through
scripts/run_safe_pytest.sh (device lock, profiler, post-run reset). The report path drives the
full capture -> tt-npe -> aggregate -> HTML pipeline in noc_report.py. Run from the repo root
with the Python env active.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.noc_placement")
    ap.add_argument("--report", action="store_true", help="Regenerate noc_placement_matrix.html (capture+tt-npe+HTML).")
    ap.add_argument(
        "--shape", default="1024,2048", help="H,W of the bf16 tiled tensor (tile-aligned). Default 1024,2048."
    )
    ap.add_argument("--cores", type=int, default=8, help="line length (<= min grid dim). Default 8.")
    ap.add_argument(
        "--block", type=int, default=16, help="pages issued per NoC barrier (in-flight pressure). Default 16."
    )
    ap.add_argument("--kernel-iters", type=int, default=8, help="in-kernel repeat for steady state. Default 8.")
    ap.add_argument("--iters", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    if args.report:
        from ttnn.operations.examples.noc_placement import noc_report

        return noc_report.main([])

    env = dict(
        os.environ,
        NP_SHAPE=args.shape,
        NP_CORES=str(args.cores),
        NP_BLOCK=str(args.block),
        NP_ITERS=str(args.kernel_iters),
        NP_PROFILE_ITERS=str(args.iters),
    )
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(
        f"[noc_placement] shape={args.shape} cores={args.cores} block={args.block} "
        f"kernel_iters={args.kernel_iters} iters={args.iters}"
    )
    print(f"[noc_placement] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
