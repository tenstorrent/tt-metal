# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 — the in-op A/B for the WHOLE-BLOCK cross-core gather.

`LEVERS["gather_coalesce"] = 0` restores the per-row gather AND the pre-Perf-2
blocking preference (the OFF arm falls back to `derive_shard_blocking`, and the
read-transfer gate sees the per-ROW size again), so the counterfactual covers
both halves of the change.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_ab_gather_coalesce.py
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import pytest
import ttnn

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))
import _bench_tilize as B  # noqa: E402

_ARMS = [(1, "on"), (0, "off")]


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_reshard(device, on, name):
    """The focus gather: W x2 -> H x8, 256 B source pages."""
    shape, src, dst = B.RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        levers=dict(gather_coalesce=on),
        label=f"ab_gather/reshard/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_reshard_gated(device, on, name):
    """W x4 -> H x8, 128 B source pages — the plan the read-transfer gate used to
    push onto the full grid. A coalesced block moves tile_h * 128 B, so the gate's
    premise (a small PER-ROW transfer) no longer holds there."""
    shape, src, dst = B.GATED_RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        levers=dict(gather_coalesce=on),
        label=f"ab_gather/reshard_gated/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_reshard_fp32(device, on, name):
    shape, src, dst = B.RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.float32,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        levers=dict(gather_coalesce=on),
        label=f"ab_gather/reshard_fp32/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_padded_gather(device, on, name):
    """The R_PAD twin: a padded target fed from a narrow-page source. The ragged
    tile-row falls back to the per-row read + fill inside the same kernel."""
    shape, target, cores = B.PAD_SHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, 2),
        out_mem_config=B._height_shard(target, cores),
        pad=dict(output_padded_shape=target, pad_value=3.5),
        levers=dict(gather_coalesce=on),
        label=f"ab_gather/padded/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_block_source_small(device, on, name):
    """The small BLOCK-sharded source the op's unit tests pin the placement of
    ([1,1,128,128], 2x2 block shards of 64x64 = 128 B pages). Coalescing makes the
    read-transfer gate's premise false here too, so the destination now stays
    resident — this row is why that behaviour change is measured rather than
    inherited from the bigger plan."""
    shape = [1, 1, 128, 128]
    in_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            (64, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out_cfg = B._height_shard(shape, 4)
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=in_cfg,
        out_mem_config=out_cfg,
        levers=dict(gather_coalesce=on),
        label=f"ab_gather/block_small/{name}",
    )
