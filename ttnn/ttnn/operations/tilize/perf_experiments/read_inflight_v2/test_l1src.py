# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight_v2 — the ONE topology where issue-ahead REGRESSED.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight_v2/test_l1src.py

`test_xact.py` case B measured issue-ahead LOSING on a source that lives in
another core's L1 (an L1 height-sharded input resharded to a differently-split
L1 height-sharded output: R_ALIGNED / W_REGION / P_ACCESSOR, the destination
shard winning the placement). Every regime where issue-ahead WON reads DRAM.

This file confirms that with repeats and separates the two things the candidate
changes at once:

  ahead0_d2  — the raw loop at the baseline's own cadence (raw-vs-helper control)
  ahead0_d3  — a DEEPER CB and nothing else (no transaction ids at all)
  trid_d2    — the op's existing two-slot B8 double-issue
  ahead1_d3  — the candidate (issue-ahead 1 + depth 3)
  *_coal32   — one transfer per BLOCK instead of per stick, which is
               address-legal here because a shard is one contiguous L1 range

so the report can say WHICH half of the candidate the L1 source dislikes.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest


def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.read_inflight_v2 import descriptor as D
from ttnn.operations.tilize.perf_experiments.read_inflight_v2.test_schedule import read_kernel_ns

# (label, shape, source shard cores, destination shard cores)
CELLS = {
    "src2": ([1, 1, 2048, 256], 2, 8),
    "src4": ([1, 1, 2048, 256], 4, 8),
    "tall_src4": ([1, 1, 4096, 256], 4, 8),
}

ARMS = {
    "baseline_helper": dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2, coalesce=1),
    "ahead0_d2": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=0, coalesce=1),
    "ahead0_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=0, coalesce=1),
    "trid_d2": dict(variant=D.VARIANT_TRID, nt_blk=1, cb_depth=2, coalesce=1),
    "ahead1_d2": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=1, coalesce=1),
    "ahead1_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=1),
    "ahead1_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=1, coalesce=1),
    "ahead0_d2_coal32": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=0, coalesce=32),
    "ahead1_d3_coal32": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=32),
}


def _shard_mem(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2] // num_cores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("rep", [0, 1, 2])
@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("cell", list(CELLS))
def test_l1_source(device, cell, arm, rep):
    shape, src_cores, dst_cores = CELLS[cell]
    torch_in = torch.randn(shape).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=_shard_mem(shape, src_cores),
    )
    mem = _shard_mem(shape, dst_cores)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mem.memory_layout, mem.shard_spec, mem.buffer_type
        ),
        device,
    )
    plan = D.plan_height_sharded(shape, ttnn.bfloat16, dst_cores)

    desc = D.build(tt_in, tt_out, plan, **ARMS[arm])
    ttnn.generic_op([tt_in, tt_out], desc)
    ttnn.synchronize_device(device)
    read_kernel_ns(device)
    out = ttnn.generic_op([tt_in, tt_out], desc)
    ttnn.synchronize_device(device)
    ns = read_kernel_ns(device)

    exact = torch.equal(ttnn.to_torch(out), torch_in)
    logger.info(f"RIV2L1SRC rep{rep}/{cell}/{arm}: ns={ns} bit_exact={exact} blocks/core={plan.blocks_per_core}")
    assert exact, f"{cell}/{arm}: NOT bit-exact — arm disqualified"
