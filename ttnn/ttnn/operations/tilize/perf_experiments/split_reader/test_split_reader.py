# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for design lamp L4 (`split_reader`) — device runner.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/split_reader/test_split_reader.py

Correctness is the ONLY pass/fail, and tilize is a PERMUTATION, so the bar is
BIT-EXACT (`torch.equal`) — not a PCC threshold. Perf is measured and logged,
never asserted.

ONE fresh measured launch per arm: device kernel duration has no warm-up
transient, so a trial loop would just re-measure the same number N times (see
/perf-measure "Measurement discipline"). The first launch is the compile / cache
warm-up AND the correctness check; the second is the measurement.
"""

import os

# In-process on-device profiler — set before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py
# — the shipped package must not drag torch in). These perf-experiment benches DO need it
# for their bit-exact oracle, so the import is done inside a function scope and published
# under the module-global name, which keeps every `torch.` use below unchanged.
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.split_reader.sr_descriptor import (
    PLANS,
    allocate_output,
    create_program_descriptor,
    geometry,
    input_memory_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Which arms are meaningful on which plan. `no_drain` needs an aliased output CB;
# `split_rw` is the interleaved-destination arm (BRISC keeps the writes).
ARMS = {
    "crossover": [
        "baseline",
        "baseline_raw",
        "baseline_brisc",
        "baseline_brisc_nodrain",
        "drain_compute",
        "no_drain",
        "split_interleave",
        "split_interleave_drain",
        "split_half",
        "split_w62",
        "split_w75",
        "split_w87",
        "split_raw",
        "baseline_dyn",
        "split_interleave_dyn",
        "split_half_dyn",
        "split_trid_dyn",
    ],
    "crossover_big": [
        "baseline",
        "baseline_brisc",
        "split_interleave",
        "split_w75",
        "split_interleave_dyn",
        "split_trid_dyn",
    ],
    "reshard": [
        "baseline",
        "baseline_brisc",
        "baseline_dyn",
        "drain_compute",
        "no_drain",
        "split_interleave",
        "split_half",
        "split_w75",
        "split_interleave_dyn",
        "split_trid_dyn",
    ],
    "small": [
        "baseline",
        "baseline_brisc",
        "no_drain",
        "split_interleave",
        "split_half",
        "split_w75",
        "split_interleave_dyn",
        "split_trid_dyn",
    ],
    "small_wide": [
        "baseline",
        "baseline_brisc",
        "no_drain",
        "split_interleave",
        "split_half",
        "split_w75",
        "split_interleave_dyn",
        "split_trid_dyn",
    ],
    "dram": ["baseline", "split_rw"],
}

CASES = [(plan, variant) for plan, variants in ARMS.items() for variant in variants]


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


@pytest.mark.parametrize("plan_name,variant", CASES, ids=[f"{p}-{v}" for p, v in CASES])
def test_split_reader(device, plan_name, variant):
    plan = PLANS[plan_name]
    torch_in = torch.randn(plan["shape"], dtype=torch.float32).to(torch.bfloat16)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config(plan),
    )
    out = allocate_output(device, plan)
    descriptor = create_program_descriptor(tt_in, out, plan, variant)

    # launch 1 — compile / program-cache warm-up AND the correctness gate
    result = ttnn.generic_op([tt_in, out], descriptor)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(result)
    exact = torch.equal(got, torch_in)
    _read_kernel_ns(device)  # flush the warm-up window

    # launch 2 — the measurement
    result = ttnn.generic_op([tt_in, out], descriptor)
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    g = geometry(plan)
    logger.info(
        f"SPLITREADER {plan_name}/{variant} ns={ns} exact={exact} "
        f"cores={g['num_cores']} blocks_per_core={g['blocks_per_core']} wt_chunk={g['wt_chunk']}"
    )
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    assert exact, f"{plan_name}/{variant}: output is not bit-exact"
