# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight_v2 sub-question — the read TRANSACTION SIZE.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight_v2/test_xact.py

Perf 1's read-floor probe showed the same 128 KB/core moves in 12,089 ns as
512 B pages but 5,627 ns as 2 KB pages — a 2.1x lever, bigger than anything
graduated. This file decides, per SOURCE PLACEMENT, whether the reader can
actually issue a bigger transfer, and correctness is what decides it.

A: DRAM-INTERLEAVED source. One row-major stick is one interleaved page, and
   consecutive pages round-robin over the DRAM banks, so `sticks k .. k+C-1`
   are NOT contiguous behind one address. The C-stick transfer is issued anyway
   and the bit-exact gate rules. (The source tensor is allocated with a spare
   tail so the widest transfer still lands inside the buffer — the point is the
   ADDRESS MAP, not an out-of-bounds read.)

B: L1-SHARDED source whose shard spans the WHOLE tensor row. A shard is one
   contiguous L1 buffer, and with `src_row_pages == 1` its pages are whole
   sticks laid end to end, so a run of sticks INSIDE one shard is one address
   range. This is a real, reachable op cell: an L1 height-sharded input reshard
   to a differently-split L1 height-sharded output takes R_ALIGNED / W_REGION /
   P_ACCESSOR (the destination shard wins the placement), and a destination
   shard sits inside exactly one source shard whenever the source is the
   coarser split.

C: WIDE SOURCE ROW. When the source stick is ALREADY >= 2 KB the reader does
   not need to merge anything — it needs the host to stop CHOPPING. On
   `[1,1,32,16384]` the op splits W into 64 chunks to fill the grid, which
   turns a 32 KB stick into a 512 B transfer. This sweeps that split.
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
from ttnn.operations.tilize.perf_experiments.read_inflight_v2.test_schedule import (
    measure,
    read_kernel_ns,
)

SHAPE = [1, 1, 2048, 256]  # the crossover's own geometry
DST_CORES = 8
COAL_MAX = 32  # a whole block (TILE_H sticks) in one transfer


def _h_shard(shape, num_cores, layout, dtype, rows=None):
    rows = rows if rows else shape[-2]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (rows // num_cores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dst(device, shape, dtype, num_cores):
    mem = _h_shard(shape, num_cores, ttnn.TILE_LAYOUT, dtype)
    spec = ttnn.TensorSpec(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mem.memory_layout, mem.shard_spec, mem.buffer_type
    )
    return ttnn.allocate_tensor_on_device(spec, device)


def _run(device, label, tt_in, tt_out, plan, ref, **kw):
    desc = D.build(tt_in, tt_out, plan, **kw)
    ttnn.generic_op([tt_in, tt_out], desc)
    ttnn.synchronize_device(device)
    read_kernel_ns(device)
    out = ttnn.generic_op([tt_in, tt_out], desc)
    ttnn.synchronize_device(device)
    ns = read_kernel_ns(device)
    exact = torch.equal(ttnn.to_torch(out), ref)
    logger.info(
        f"RIV2XACT {label}: ns={ns} bit_exact={exact} row_bytes={plan.row_bytes} coalesce={kw.get('coalesce',1)}"
    )
    return ns, exact


# ---------------------------------------------------------------------------
# A. DRAM-interleaved source — is a multi-stick transfer address-correct?
@pytest.mark.parametrize("coal", [1, 2, 4, COAL_MAX], ids=lambda c: f"coal{c}")
def test_a_dram_interleaved_coalesce(device, coal):
    pad_rows = SHAPE[-2] + COAL_MAX * D.TILE_H  # spare tail: no transfer leaves the buffer
    torch_full = torch.randn([1, 1, pad_rows, SHAPE[-1]]).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_full,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = _dst(device, SHAPE, ttnn.bfloat16, DST_CORES)
    plan = D.plan_height_sharded(SHAPE, ttnn.bfloat16, DST_CORES)
    ref = torch_full[:, :, : SHAPE[-2], :]
    ns, exact = _run(
        device,
        f"A/dram_interleaved/coal{coal}",
        tt_in,
        tt_out,
        plan,
        ref,
        variant=D.VARIANT_AHEAD,
        nt_blk=1,
        cb_depth=3,
        ahead=1,
        coalesce=coal,
    )
    if coal == 1:
        assert exact, "the one-stick-per-transfer control must be bit-exact"
    else:
        # NOT an assertion about speed: this records whether the address map
        # allows the merge at all. A DRAM-interleaved source is expected to fail
        # it — that is what makes the lever INEXPRESSIBLE here.
        logger.info(f"A/dram_interleaved/coal{coal}: address-merge legal = {exact}")


# ---------------------------------------------------------------------------
# B. L1-sharded source, shard spans the whole row — the merge IS expressible.
_B_ARMS = {
    "baseline_helper": dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2, coalesce=1),
    "ahead1_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=1),
    "ahead1_d3_coal4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=4),
    "ahead1_d3_coal8": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=8),
    "ahead1_d3_coal32": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1, coalesce=32),
    "ahead0_d2_coal32": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=0, coalesce=32),
}


@pytest.mark.parametrize("src_cores", [2, 4], ids=lambda c: f"src{c}")
@pytest.mark.parametrize("arm", list(_B_ARMS))
def test_b_l1_shard_source_coalesce(device, arm, src_cores):
    torch_in = torch.randn(SHAPE).to(torch.bfloat16)
    mem = _h_shard(SHAPE, src_cores, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
    )
    tt_out = _dst(device, SHAPE, ttnn.bfloat16, DST_CORES)
    plan = D.plan_height_sharded(SHAPE, ttnn.bfloat16, DST_CORES)
    ns, exact = _run(device, f"B/l1shard{src_cores}/{arm}", tt_in, tt_out, plan, torch_in, **_B_ARMS[arm])
    assert exact, f"B/{arm}: NOT bit-exact — arm disqualified"


# ---------------------------------------------------------------------------
# C. A source row that is ALREADY wide: the transfer size is the HOST's W split.
_C = dict(kind="inter", shape=[1, 1, 32, 16384], grid=None, n_chunks=None)


@pytest.mark.parametrize(
    # n_chunks == 4 (WT_CHUNK 128) is omitted: its CBs alone are 1 MiB at depth 2,
    # so depth 3 would be over the op's own L1 budget and the two arms could not
    # be compared at the same blocking.
    "n_chunks,gx,gy",
    [(64, 8, 8), (32, 8, 4), (16, 8, 2), (8, 8, 1)],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("arm", ["baseline_helper", "ahead1_d3"])
def test_c_wide_row_split(device, arm, n_chunks, gx, gy):
    cfg = dict(_C, grid=(gx, gy), n_chunks=n_chunks)
    kw = (
        dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2)
        if arm == "baseline_helper"
        else dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1)
    )
    measure(device, cfg, ttnn.bfloat16, f"C/wide_row/nchunks{n_chunks}_cores{gx*gy}/{arm}", **kw)
