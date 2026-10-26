# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the four read-coalescing × bank-affinity combinations on width-sharded DRAM."""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
import tempfile
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_read_coalescing.py::test_read_coalescing_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.read_coalescing")
    ap.add_argument(
        "--cases",
        default="auto",
        help="comma list of HxWxSHARD_WIDTH (width-sharded DRAM, shard width in elements, one shard per bank), "
        "or auto (default): cases sized to the grid.",
    )
    ap.add_argument(
        "--variant",
        default="all",
        help="all, or a comma list of none,coalesce,affinity,both. Default all.",
    )
    ap.add_argument("--kernel-iters", type=int, default=1, help="in-kernel repeat. 1 = per-launch latency. Default 1.")
    ap.add_argument("--trials", type=int, default=5, help="trials per variant (median reported). Default 5.")
    ap.add_argument("--launches", type=int, default=10, help="launches averaged per trial. Default 10.")
    ap.add_argument("--report", default=None, help="also write the table to this file.")
    args = ap.parse_args()

    env = dict(
        os.environ,
        RC_CASES=args.cases,
        RC_VARIANT=args.variant,
        RC_ITERS=str(args.kernel_iters),
        RC_TRIALS=str(args.trials),
        RC_LAUNCHES=str(args.launches),
    )
    # pytest captures the logger, so the table always goes through a file and is printed here.
    if args.trials < 1 or args.launches < 1 or args.kernel_iters < 1:
        ap.error("trials, launches and kernel-iters must be positive")
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", _TEST]
    print(f"[read_coalescing] cases={args.cases} variant={args.variant} kernel_iters={args.kernel_iters}")
    print(f"[read_coalescing] {' '.join(cmd)}")
    # Always use a fresh report: a failed run must never display or overwrite an older result.
    with tempfile.TemporaryDirectory(prefix="read_coalescing_") as tmp_dir:
        report = Path(tmp_dir) / "report.txt"
        env["RC_REPORT"] = str(report)
        with tempfile.TemporaryFile(mode="w+") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0 or not report.is_file() or report.stat().st_size == 0:
                log.seek(0)
                shutil.copyfileobj(log, sys.stderr)
                if result.returncode != 0:
                    print(f"[read_coalescing] benchmark failed (exit {result.returncode})", file=sys.stderr)
                    return result.returncode
                print("[read_coalescing] benchmark produced no report", file=sys.stderr)
                return 1
        text = report.read_text()
        if args.report:
            Path(args.report).write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
