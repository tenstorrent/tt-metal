# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for `split_reader_v2` — device runner.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/split_reader_v2/test_sr2.py \
        -k "focus"

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


# `ttnn/` may not import torch at module scope (the shipped package must not drag
# torch in), so the import is done inside a function and published as a global.
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.split_reader_v2.sr2_descriptor import (
    PLANS,
    allocate_output,
    cb_l1_bytes,
    create_program_descriptor,
    geometry,
    input_memory_config,
    plan_dtype,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# --- arm sets --------------------------------------------------------------
# FOCUS: the full decomposition on the two plans the coordinator flagged.
#   op_baseline -> nodrain     : what freeing BRISC of the drain alone buys
#   nodrain     -> alt_tax     : what the compute-side alternation COSTS
#   alt_tax     -> split_*     : what moving half the reads to BRISC buys
FOCUS_ARMS = [
    "op_baseline",
    "nodrain",
    "alt_tax",
    "cdrain",
    "raw_baseline",
    "split_il",
    "split_il_bdrain",
    "split_il_cdrain",
    "split_p75",
    "split_p67",
    "split_p60",
    "split_p57",
    "split_p67_cdrain",
    "split_w75",
    "split_trid",
    "split_trid_p75",
]
# RATIO: the dedicated-NoC issue-ratio sweep alone (a second session, so the
# focus decomposition above stays one self-consistent block of numbers).
RATIO_ARMS = ["op_baseline", "split_il", "split_p57", "split_p60", "split_p67", "split_p75"]
# DOMAIN: baseline + the candidate flavors, everywhere else.
DOMAIN_ARMS = ["op_baseline", "split_il", "split_p67", "split_trid"]
# SHIP: the two integration forms (compute drains, so the aliased output CB keeps
# exactly one consumer) against the honest baseline.
SHIP_ARMS = ["op_baseline", "split_trid_cdrain", "split_il_cdrain"]
SHIP_PLANS = [
    "crossover",
    "crossover_big",
    "crossover_tall",
    "crossover_512",
    "reshard",
    "reshard_fp32",
    "reshard_wide",
    "gather_h",
]
# INTERLEAVED DESTINATION: the measured exclusion.
DRAM_ARMS = ["op_baseline", "split_rw", "split_rw_p88"]

FOCUS_PLANS = ["crossover", "reshard"]
DOMAIN_PLANS = [
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
]
DRAM_PLANS = ["dram", "dram_small"]

CASES = (
    [("focus", p, v) for p in FOCUS_PLANS for v in FOCUS_ARMS]
    + [("ratio", p, v) for p in FOCUS_PLANS for v in RATIO_ARMS]
    + [("domain", p, v) for p in DOMAIN_PLANS for v in DOMAIN_ARMS]
    + [("excl", p, v) for p in DRAM_PLANS for v in DRAM_ARMS]
    + [("ship", p, v) for p in SHIP_PLANS for v in SHIP_ARMS]
)


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


@pytest.mark.parametrize("group,plan_name,variant", CASES, ids=[f"{g}-{p}-{v}" for g, p, v in CASES])
def test_sr2(device, group, plan_name, variant):
    plan = PLANS[plan_name]
    dtype = plan_dtype(plan)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_in = torch.randn(plan["shape"], dtype=torch.float32).to(torch_dtype)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config(plan),
    )
    out = allocate_output(device, plan)
    descriptor = create_program_descriptor(tt_in, out, plan, variant)

    # launch 1 — compile / program-cache warm-up AND the correctness gate
    try:
        result = ttnn.generic_op([tt_in, out], descriptor)
    except RuntimeError as exc:
        # An arm whose CBs do not FIT is a real, recordable result (the split's
        # second input CB is L1 the op does not have on a wide shard), not a test
        # failure. Anything else re-raises.
        text = str(exc)
        if "circular buffer" not in text and "Out of Memory" not in text:
            raise
        logger.info(
            f"SR2 {plan_name}/{variant} ns=L1_OOM exact=n/a "
            f"in_cb_l1={cb_l1_bytes(plan, variant)} :: {text.splitlines()[0][:160]}"
        )
        return
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
        f"SR2 {plan_name}/{variant} ns={ns} exact={exact} "
        f"cores={g['num_cores']} blocks_per_core={g['blocks_per_core']} "
        f"wt_chunk={g['wt_chunk']} in_cb_l1={cb_l1_bytes(plan, variant)}"
    )
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    assert exact, f"{plan_name}/{variant}: output is not bit-exact"
