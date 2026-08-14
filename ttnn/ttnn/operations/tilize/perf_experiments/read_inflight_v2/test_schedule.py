# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight_v2 — issue-ahead as the ONE R_ALIGNED reader loop.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight_v2/test_schedule.py

Correctness is the ONLY pass/fail and tilize is a PERMUTATION, so the bar is
bit-exact (`torch.equal`). Perf is measured and printed, never asserted.

One fresh-cache warm launch (compile + program cache) then ONE measured launch
per arm — device kernel duration has no warm-up transient.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py).
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.read_inflight_v2 import descriptor as D

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# ---------------------------------------------------------------------------
# REGIMES — the domain sweep. `kind`:
#   "shard"  DRAM/L1 ROW_MAJOR source -> HEIGHT-sharded L1 TILE destination
#            (the op's R_ALIGNED / W_REGION / n_chunks == 1 path).
#   "inter"  interleaved -> interleaved TILE (the op's R_ALIGNED / W_BLOCKS path,
#            which is where the existing B8 trid branch is host-enabled).
REGIMES = {
    # the perf-flagged DRAM -> local-L1-shard crossover (8 blocks/core, 512 B reads)
    "focus": dict(kind="shard", shape=[1, 1, 2048, 256], num_cores=8),
    # same topology, 4x taller shard (32 blocks/core) — separates "small lever"
    # from "focus plan is tail-dominated"
    "tall": dict(kind="shard", shape=[1, 1, 8192, 256], num_cores=8),
    # small crossover: per-core-overhead regime, 128 B transfers (4 blocks/core)
    "small": dict(kind="shard", shape=[1, 1, 512, 64], num_cores=4),
    # ONE block per core — the issue-ahead window cannot fill. Expected FLAT.
    "oneblock": dict(kind="shard", shape=[1, 1, 256, 256], num_cores=8),
    # THE SMALLEST REGIME: 2 tiles on 2 cores, 64 B transfers, one block each.
    # A deeper pipeline can only LOSE here — that is why it is measured.
    "smallest": dict(kind="inter", shape=[1, 1, 32, 64], grid=(2, 1), n_chunks=2),
    # wide/short: one block per core over the full grid, 512 B transfers
    "wide_short": dict(kind="inter", shape=[1, 1, 32, 16384], grid=(8, 8), n_chunks=64),
    # 64-core interleaved square — already at the measured DRAM-copy floor
    "interleaved": dict(kind="inter", shape=[1, 1, 2048, 2048], grid=(8, 8), n_chunks=4),
    # W_BLOCKS interleaved on EIGHT cores: the topology the op host-enables the
    # existing B8 trid branch on, but NOT fabric-saturated — this is the cell that
    # decides whether the generalized loop can replace the B8 special case.
    "wblocks8": dict(kind="inter", shape=[1, 1, 2048, 256], grid=(8, 1), n_chunks=1),
}


def _height_shard_cfg(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2] // num_cores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


def read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def torch_source(shape, dtype):
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype == ttnn.uint16:
        return torch.randint(0, 65536, shape, dtype=torch.int32).to(torch.int32)
    return torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)


def make(device, cfg, dtype, *, n_chunks=None):
    shape = cfg["shape"]
    torch_in = torch_source(shape, dtype)
    tt_in = ttnn.from_torch(
        torch_in, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if cfg["kind"] == "shard":
        mem = _height_shard_cfg(shape, cfg["num_cores"])
        spec = ttnn.TensorSpec(
            ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mem.memory_layout, mem.shard_spec, mem.buffer_type
        )
        plan = D.plan_height_sharded(shape, dtype, cfg["num_cores"])
    else:
        spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
        plan = D.plan_interleaved(shape, dtype, cfg["grid"], n_chunks if n_chunks else cfg["n_chunks"])
    tt_out = ttnn.allocate_tensor_on_device(spec, device)
    return torch_in, tt_in, tt_out, plan


def run_arm(device, cfg, dtype, label, *, variant, nt_blk, cb_depth, ahead=1, coalesce=1, n_chunks=None):
    torch_in, tt_in, tt_out, plan = make(device, cfg, dtype, n_chunks=n_chunks)
    desc = D.build(
        tt_in, tt_out, plan, variant=variant, nt_blk=nt_blk, cb_depth=cb_depth, ahead=ahead, coalesce=coalesce
    )

    ttnn.generic_op([tt_in, tt_out], desc)  # warm: compile + program cache
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # flush the warm-up window

    out = ttnn.generic_op([tt_in, tt_out], desc)  # the ONE measured launch
    ttnn.synchronize_device(device)
    ns = read_kernel_ns(device)

    got = ttnn.to_torch(out)
    exact = torch.equal(got, torch_in)
    l1_in = plan.in_cb_bytes(nt_blk, cb_depth)
    l1_out = plan.out_cb_bytes(2)
    logger.info(
        f"RIV2 {label}: ns={ns} bit_exact={exact} "
        f"in_cb={l1_in}B out_cb={l1_out}B total_cb={l1_in + l1_out}B (budget {D.CB_L1_BUDGET}B) "
        f"blocks/core={plan.blocks_per_core} wt_chunk={plan.wt_chunk} row_bytes={plan.row_bytes} "
        f"cores={len(plan.cores)}"
    )
    assert ns is not None, "profiler produced no data"
    return ns, exact, l1_in + l1_out


def measure(device, cfg, dtype, label, **kw):
    ns, exact, l1 = run_arm(device, cfg, dtype, label, **kw)
    assert exact, f"{label}: NOT bit-exact — arm disqualified"
    assert l1 <= D.CB_L1_BUDGET, f"{label}: {l1} B over the op's CB L1 budget"
    return ns


# ---------------------------------------------------------------------------
# ARMS. `baseline_helper` is the op's current approach for this part, verbatim;
# `trid_d2` is the op's EXISTING B8 special case; every `aheadN_dM` is the ONE
# generalized loop at (issue-ahead N, CB depth M).
ARMS = {
    "baseline_helper": dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2),
    "trid_d2": dict(variant=D.VARIANT_TRID, nt_blk=1, cb_depth=2),
    # ahead == 0 is the generalized loop degenerating to the plain
    # barrier-per-block loop: the control that separates "raw vs helper" from
    # "issue-ahead".
    "ahead0_d2": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=0),
    # ahead == 0 at depth 3/4 isolates the DEEPER CB from the issue-ahead: same
    # plain barrier-per-block schedule, more slack behind it. If the win came
    # from the CB alone these would carry it.
    "ahead0_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=0),
    "ahead0_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=0),
    # ahead == 1, depth == 2 IS the B8 schedule expressed through the ONE loop.
    "ahead1_d2": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=1),
    "ahead1_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1),
    "ahead1_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=1),
    "ahead2_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=2),
    "ahead2_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=2),
    "ahead3_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=3),
    "ahead3_d5": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=5, ahead=3),
    "ahead4_d5": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=5, ahead=4),
    "ahead4_d6": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=6, ahead=4),
}


def _blocks_per_core(regime):
    cfg = REGIMES[regime]
    if cfg["kind"] == "shard":
        return cfg["shape"][-2] // cfg["num_cores"] // D.TILE_H
    return (cfg["shape"][-2] // D.TILE_H) * cfg["n_chunks"] // (cfg["grid"][0] * cfg["grid"][1])


def _skip_if_redundant(regime, arm_name, arm):
    per_core = _blocks_per_core(regime)
    if arm["variant"] == D.VARIANT_TRID and per_core < 2:
        pytest.skip("the two-slot B8 form needs at least two groups")
    # A window wider than the work degenerates to a shallower one: the drain loop
    # simply pushes whatever is outstanding. ahead == 1 is always kept so every
    # regime has the recommended arm.
    if arm["variant"] == D.VARIANT_AHEAD and arm["ahead"] > 1 and arm["ahead"] >= per_core:
        pytest.skip(f"ahead={arm['ahead']} degenerates at {per_core} blocks/core")


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("regime", list(REGIMES))
def test_arm(device, regime, arm):
    _skip_if_redundant(regime, arm, ARMS[arm])
    measure(device, REGIMES[regime], ttnn.bfloat16, f"{regime}/bf16/{arm}", **ARMS[arm])


# --- domain: dtype ---------------------------------------------------------
# fp32 doubles the transfer and runs under the op's fp32 exactness contract
# (fp32 DEST + lossless unpack), applied IDENTICALLY to every arm. uint8 halves
# it. Bit-exactness is still the pass bar for both.
_DTYPE_ARMS = ["baseline_helper", "trid_d2", "ahead1_d2", "ahead1_d3", "ahead1_d4", "ahead2_d4"]


@pytest.mark.parametrize("arm", _DTYPE_ARMS)
@pytest.mark.parametrize("regime", ["focus", "small", "wblocks8"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.uint8], ids=["fp32", "uint8"])
def test_arm_dtype(device, dtype, regime, arm):
    _skip_if_redundant(regime, arm, ARMS[arm])
    measure(device, REGIMES[regime], dtype, f"{regime}/{dtype}/{arm}", **ARMS[arm])


# --- repeat the headline arms so the win/null call is not on noise ----------
@pytest.mark.parametrize("rep", [0, 1, 2])
@pytest.mark.parametrize("arm", ["baseline_helper", "trid_d2", "ahead1_d3", "ahead2_d4"])
@pytest.mark.parametrize("regime", ["focus", "wblocks8", "smallest"])
def test_repeat(device, regime, arm, rep):
    _skip_if_redundant(regime, arm, ARMS[arm])
    measure(device, REGIMES[regime], ttnn.bfloat16, f"rep{rep}/{regime}/{arm}", **ARMS[arm])


# The 64-core square sits at the measured DRAM-copy floor, so its arms land
# inside the noise band and the flat/win call has to be repeated to be honest.
@pytest.mark.parametrize("rep", [0, 1, 2])
@pytest.mark.parametrize("arm", ["baseline_helper", "ahead0_d2", "ahead1_d3", "ahead1_d4"])
def test_repeat_interleaved(device, arm, rep):
    measure(device, REGIMES["interleaved"], ttnn.bfloat16, f"rep{rep}/interleaved/{arm}", **ARMS[arm])
