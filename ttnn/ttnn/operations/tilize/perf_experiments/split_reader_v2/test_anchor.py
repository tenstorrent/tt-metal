# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ANCHOR: the REAL tilize op on this experiment's plans.

The bake-off's `op_baseline` arm is a RECONSTRUCTION of today's op scheme, not
the op itself. This file measures the op itself on the same plans through the
same profiler, so the reconstruction can be pinned against it — a bake-off whose
baseline is faster or slower than the thing it claims to reconstruct is measuring
something else.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/split_reader_v2/test_anchor.py

The op is NOT modified: this imports and calls it exactly as a user would.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize import tilize

from ttnn.operations.tilize.perf_experiments.split_reader_v2.sr2_descriptor import (
    PLANS,
    geometry,
    input_memory_config,
    output_memory_config,
    plan_dtype,
)


# `ttnn/` may not import torch at module scope
# (scripts/validate_no_global_torch_imports.py), so it is imported lazily.
def _torch():
    import torch

    return torch


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

ANCHOR_PLANS = [
    "crossover",
    "reshard",
    "crossover_512",
    "crossover_2048",
    "reshard_wide",
    "gather_h",
    "crossover_big",
    "crossover_wide",
    "crossover_fp32",
    "crossover_1blk",
    "crossover_tall",
    "reshard_w4",
    "reshard_fp32",
    "small",
    "small_wide",
    "dram",
    "dram_small",
]


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


@pytest.mark.parametrize("plan_name", ANCHOR_PLANS)
def test_anchor_real_op(device, plan_name):
    plan = PLANS[plan_name]
    dtype = plan_dtype(plan)
    torch_dtype = _torch().bfloat16 if dtype == ttnn.bfloat16 else _torch().float32
    torch_in = _torch().randn(plan["shape"], dtype=_torch().float32).to(torch_dtype)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config(plan),
    )
    out_cfg = output_memory_config(plan)

    tilize(tt_in, memory_config=out_cfg)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window

    got = tilize(tt_in, memory_config=out_cfg)
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    exact = _torch().equal(ttnn.to_torch(got), torch_in)
    g = geometry(plan)
    logger.info(
        f"ANCHOR {plan_name} ns={ns} exact={exact} "
        f"bench_cores={g['num_cores']} bench_blocks_per_core={g['blocks_per_core']} "
        f"bench_wt_chunk={g['wt_chunk']}"
    )
    assert ns is not None
    assert exact
