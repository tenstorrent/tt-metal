# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: re-measure via_row_major vs face_rows vs runs on your own shapes and tile heights.

    python -m ttnn.operations.examples.retile_face_regroup [--cells "H:W:th_in:th_out,..."] [--trials N] [--report PATH]

Runs the device-perf sweep of tests/ttnn/unit_tests/operations/examples/test_retile_face_regroup.py through
scripts/run_safe_pytest.sh (device lock, in-process profiler). Without --cells it runs the predefined sweep.
"""

import argparse
import os
import subprocess
import sys

_TEST = "tests/ttnn/unit_tests/operations/examples/test_retile_face_regroup.py::test_retile_face_regroup_device_perf"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.retile_face_regroup")
    ap.add_argument("--cells", help='comma-separated "H:W:th_in:th_out" cells (default: predefined sweep)')
    ap.add_argument("--trials", type=int, default=3, help="measured passes (median); default 3")
    ap.add_argument("--report", default="", help="also write the Markdown report here")
    args = ap.parse_args()
    env = dict(os.environ, RFR_SWEEPS="1", RFR_TRIALS=str(args.trials), RFR_REPORT=args.report)
    env.pop("RFR_CELLS", None)
    if args.cells:
        env["RFR_CELLS"] = args.cells
    return subprocess.call(["scripts/run_safe_pytest.sh", "--run-all", _TEST], env=env)


if __name__ == "__main__":
    sys.exit(main())
