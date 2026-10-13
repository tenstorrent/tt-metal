# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
import sys
from .program_descriptor_with_inline_kernels import VARIANTS


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main():
    p = argparse.ArgumentParser(description="Measure transaction sizes for DRAM to resident L1 height shards.")
    p.add_argument("--variant", nargs="+", choices=("all",) + VARIANTS, default=["all"])
    p.add_argument("--shape", nargs=4, type=positive, metavar=("N", "C", "H", "W"))
    p.add_argument("--cores", type=positive)
    p.add_argument(
        "--sweep-cores", nargs="+", help="all, or a list of model-feasible core counts; requires --model-json"
    )
    p.add_argument("--trials", type=positive, default=1)
    p.add_argument("--kernel-iters", type=positive, default=1)
    p.add_argument("--report")
    p.add_argument("--profile", action="store_true", help="Capture Tracy through the safe pytest wrapper")
    p.add_argument("--zones", action="store_true", help="Instrument read issue, barrier and total device zones")
    p.add_argument("--model-json", help="Optional shape, cores, and predicted timings snapshot")
    a = p.parse_args()
    if a.sweep_cores and (not a.model_json or a.cores):
        p.error("--sweep-cores requires --model-json and cannot be combined with --cores")
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_PROFILER_MID_RUN_DUMP="1", TT_METAL_PROFILER_CPP_POST_PROCESS="1")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "error")
    env["DRAM_L1_TRIALS"] = str(a.trials)
    env["DRAM_L1_KERNEL_ITERS"] = str(a.kernel_iters)
    env["DRAM_L1_PROFILE_ZONES"] = "1" if a.zones else "0"
    if "all" not in a.variant:
        env["DRAM_L1_VARIANTS"] = ",".join(a.variant)
    if a.shape:
        env["DRAM_L1_SHAPE"] = ",".join(map(str, a.shape))
    if a.cores:
        env["DRAM_L1_CORES"] = str(a.cores)
    if a.report:
        env["DRAM_L1_REPORT"] = a.report
    if a.model_json:
        env["DRAM_L1_MODEL_JSON"] = a.model_json
    node = "test_dram_l1_transaction_size"
    if a.sweep_cores:
        env["DRAM_L1_CORE_COUNTS"] = (
            "all" if a.sweep_cores == ["all"] else ",".join(str(positive(v)) for v in a.sweep_cores)
        )
        node = "test_dram_l1_core_sweep"
    return subprocess.call(
        [
            "scripts/run_safe_pytest.sh",
            "--no-precompile",
            *(["--profile"] if a.profile else []),
            "tests/ttnn/unit_tests/operations/examples/test_dram_l1_transaction_size.py::" + node,
            "-s",
            "-q",
        ],
        env=env,
    )


if __name__ == "__main__":
    sys.exit(main())
