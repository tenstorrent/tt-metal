# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2, Step 5 — the GUARD SET: one representative per distinct kernel path
x layout x placement, measured whole-op.

This is how a Perf-2 graduation is checked. Not a correctness gate (golden owns
that) — it is how a MATERIAL REGRESSION on a supported cell is FOUND, which is
the only thing that earns a carve-out, and how the win is confirmed end-to-end.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_guardset.py

Every row prints `BENCH tilize guard/<row> ... ns=<device kernel ns>`. One
fresh launch per row (device kernel duration has no warm-up transient).

Rows and why each is in the set:
  * a_square / c_multiblock — interleaved DRAM->DRAM at the Refinement-3 measured
    DRAM-copy floor. These must not move.
  * b_wide_short — one block per core, no read/write overlap (Perf 1's A0 target).
  * d_smallest — the per-core-overhead regime, where any deeper pipeline can lose.
  * tile_h 16/8/1 — the REGULAR (non-fast) LLK tilize path.
  * out_dtype fp32 / bfloat8_b / uint32 / uint8 — the cast and integer datapaths.
  * retile 32->8 / 32->16 / 1->32 — the R_RETILE reader.
  * widening_pad — R_PAD + out_fill (the pad stamp + the write roofline).
  * crossover / reshard / padded_local_shard / shard_same_spec — the sharded
    placements: accessor source with a local-shard destination, the cross-core L1
    gather, the padded gather, and the zero-copy both-sides plan.
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import pytest
import ttnn

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))
import _bench_tilize as B  # noqa: E402


def _m(device, row, shape, dtype, **kw):
    return B._measure(device, shape, dtype, label=f"guard/{row}", **kw)


# --- interleaved DRAM->DRAM, the roofline-gated rows ------------------------
@pytest.mark.parametrize("regime", ["a_square", "b_wide_short", "c_multiblock", "d_smallest"])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_guard_interleaved(device, regime, dtype_name):
    dtype = ttnn.bfloat16 if dtype_name == "bf16" else ttnn.float32
    _m(device, f"{regime}/{dtype_name}", B.SHAPES[regime], dtype)


# --- the REGULAR (non-fast) LLK tilize path: tiny tiles ---------------------
@pytest.mark.parametrize("tile_height", [16, 8, 1])
def test_guard_tile_height(device, tile_height):
    _m(device, f"tile_h={tile_height}/a_square", B.SHAPES["a_square"], ttnn.bfloat16, tile_h=tile_height)


@pytest.mark.parametrize("tile_height", [32, 8])
def test_guard_tile_height_smallest(device, tile_height):
    _m(device, f"tile_h={tile_height}/d_smallest", B.SHAPES["d_smallest"], ttnn.bfloat16, tile_h=tile_height)


# --- cast + integer datapaths ----------------------------------------------
@pytest.mark.parametrize(
    "in_dt,out_dt,row",
    [
        (ttnn.bfloat16, ttnn.float32, "bf16_to_fp32"),
        (ttnn.bfloat16, ttnn.bfloat8_b, "bf16_to_bf8b"),
        (ttnn.uint32, None, "uint32"),
        (ttnn.uint8, None, "uint8"),
    ],
)
def test_guard_dtype(device, in_dt, out_dt, row):
    _m(device, f"{row}/a_square", B.SHAPES["a_square"], in_dt, out_dtype=out_dt)


# --- R_RETILE ---------------------------------------------------------------
@pytest.mark.parametrize("in_tile_h,tile_height", [(32, 8), (32, 16), (1, 32)], ids=["32to8", "32to16", "1to32"])
def test_guard_retile(device, in_tile_h, tile_height):
    _m(
        device,
        f"retile/{in_tile_h}to{tile_height}",
        B._RETILE_SHAPE,
        ttnn.bfloat16,
        tile_h=tile_height,
        in_tile_h=in_tile_h,
    )


# --- R_PAD + out_fill -------------------------------------------------------
def test_guard_widening_pad(device):
    shape, target = B._OUT_FILL_SHAPE
    _m(
        device,
        "widening_pad",
        shape,
        ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=target, pad_value=10.2),
    )


# --- sharded placements -----------------------------------------------------
def test_guard_crossover(device):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    _m(device, "crossover", shape, ttnn.bfloat16, out_mem_config=B._height_shard(shape, cores))


def test_guard_reshard(device):
    shape, src, dst = B.RESHARD_SHAPE
    _m(
        device,
        "reshard",
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
    )


def test_guard_reshard_gated(device):
    shape, src, dst = B.GATED_RESHARD_SHAPE
    _m(
        device,
        "reshard_gated",
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
    )


def test_guard_padded_local_shard(device):
    shape, target, cores = B.PAD_SHARD_SHAPE
    _m(
        device,
        "padded_local_shard",
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(target, cores),
        pad=dict(output_padded_shape=target, pad_value=3.5),
    )


@pytest.mark.parametrize("tile_height", [8, 1])
def test_guard_shard_same_spec_tiny_tile(device, tile_height):
    """The zero-NoC plan (both CBs alias the resident shard) at a sub-32 tile
    height: the ONE row in the guard set where the TRISC is the critical path and
    neither DRAM leg exists. This is where a compute-side change shows undiluted."""
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    cfg = B._height_shard(shape, cores)
    _m(
        device,
        f"shard_same/tile_h={tile_height}",
        shape,
        ttnn.bfloat16,
        tile_h=tile_height,
        in_mem_config=cfg,
        out_mem_config=cfg,
    )


@pytest.mark.parametrize("regime", ["e_shard_same_small", "e_shard_same_wide"])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_guard_shard_same_spec(device, regime, dtype_name):
    shape, cores = B.SHARDED_SHAPES[regime]
    dtype = ttnn.bfloat16 if dtype_name == "bf16" else ttnn.float32
    cfg = B._height_shard(shape, cores)
    _m(device, f"shard_same/{regime}/{dtype_name}", shape, dtype, in_mem_config=cfg, out_mem_config=cfg)
