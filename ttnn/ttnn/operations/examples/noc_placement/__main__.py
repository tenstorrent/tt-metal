# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure noc_placement on YOUR shapes/params, or regenerate the report.

    # measure the read/write × NoC × placement matrix (device kernel ns/op)
    python -m ttnn.operations.examples.noc_placement [--shape H,W] [--cores N] [--block N]
                                                      [--kernel-iters N] [--iters N]

    # write split: the usual pairing vs 1/4, 1/2, all writes on the read NoC, and the swapped pairing
    python -m ttnn.operations.examples.noc_placement --write-split [--placement grid|row|column|diagonal]
                                                      [--cores N] [--shape H,W] [--block N] [--kernel-iters N]
                                                      [--in-mem dram|l1] [--out-mem dram|l1] [--trials N]

    # swap the pairing (reads NoC1 / writes NoC0) on the N DRAM-nearest worker columns only, N in
    # {0,1,2,3,4,6} ranked from bank 0's DRAM column (f<N>) or from every DRAM column (e<N>), plus "all"
    python -m ttnn.operations.examples.noc_placement --swap-columns [--placement grid|row|column|diagonal]
                                                      [--cores N] [--shape H,W] [--block N] [--kernel-iters N]
                                                      [--in-mem dram|l1] [--out-mem dram|l1] [--trials N]

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
_WS_TEST = "tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_perf"
_SW_TEST = "tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.noc_placement")
    ap.add_argument("--report", action="store_true", help="Regenerate noc_placement_matrix.html (capture+tt-npe+HTML).")
    ap.add_argument(
        "--write-split",
        action="store_true",
        help="Measure the write split (1/4, 1/2, all writes on the read NoC) and the swapped pairing on one copy.",
    )
    ap.add_argument(
        "--swap-columns",
        action="store_true",
        help="Measure swap_columns=N (reads NoC1 / writes NoC0 on the N DRAM-nearest columns) vs the usual pairing.",
    )
    ap.add_argument(
        "--placement", default="grid", help="--write-split/--swap-columns: grid|row|column|diagonal. Default grid."
    )
    ap.add_argument("--in-mem", default="dram", choices=("dram", "l1"), help="--write-split/--swap-columns: input.")
    ap.add_argument("--out-mem", default="dram", choices=("dram", "l1"), help="--write-split/--swap-columns: output.")
    ap.add_argument("--trials", type=int, default=5, help="--write-split/--swap-columns: rotated trials. Default 5.")
    ap.add_argument(
        "--shape", default="1024,2048", help="H,W of the bf16 tiled tensor (tile-aligned). Default 1024,2048."
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=None,
        help="line length (<= min grid dim). Default 8; with --write-split/--swap-columns: the whole grid.",
    )
    ap.add_argument(
        "--block", type=int, default=16, help="pages issued per NoC barrier (in-flight pressure). Default 16."
    )
    ap.add_argument(
        "--kernel-iters",
        type=int,
        default=None,
        help="in-kernel repeat for steady state. Default 8 (--write-split/--swap-columns: 1).",
    )
    ap.add_argument("--iters", type=int, default=20, help="profiled launches per case (averaged). Default 20.")
    args = ap.parse_args()

    if args.report:
        from ttnn.operations.examples.noc_placement import noc_report

        return noc_report.main([])

    if args.write_split or args.swap_columns:
        env = dict(
            os.environ,
            **({"NP_WS_CUSTOM": "1"} if args.write_split else {"NP_SWAP_CUSTOM": "1"}),
            NP_PLACEMENT=args.placement,
            NP_SHAPE=args.shape,
            NP_BLOCK=str(args.block),
            NP_ITERS=str(args.kernel_iters or 1),
            NP_IN_MEM=args.in_mem,
            NP_OUT_MEM=args.out_mem,
            NP_WS_TRIALS=str(args.trials),
            NP_SW_TRIALS=str(args.trials),
        )
        if args.cores is not None:
            env["NP_CORES"] = str(args.cores)
        cmd = ["scripts/run_safe_pytest.sh", "--run-all", _WS_TEST if args.write_split else _SW_TEST]
        print(f"[noc_placement] {' '.join(cmd)}")
        return subprocess.call(cmd, env=env)

    args.cores = 8 if args.cores is None else args.cores
    args.kernel_iters = 8 if args.kernel_iters is None else args.kernel_iters
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
