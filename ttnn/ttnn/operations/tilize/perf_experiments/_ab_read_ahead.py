# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 — the in-op A/B for the reader's issue-ahead schedule.

`LEVERS["read_ahead"] = 0` restores BOTH halves of the pre-Perf-2 state: the
plain barrier-per-block loop AND the shallower input CB (the extra slack group
is only allocated on a plan that takes the schedule, so the OFF arm gets exactly
the blocking it had before this round). That makes this a true counterfactual,
not just a kernel-branch flip.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_ab_read_ahead.py

Rows are the guard set's accessor-read cells plus the two the schedule is
expected to be FLAT on (the DRAM-bound square, the smallest 2-tile cell) — flat
is a result, not a reason to fence the pattern off those cells.
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import pytest
import ttnn

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))
import _bench_tilize as B  # noqa: E402

_ARMS = [(1, "on"), (0, "off")]


def _ab(device, row, shape, dtype, on, name, **kw):
    B._measure(device, shape, dtype, levers=dict(read_ahead=on), label=f"ab_read_ahead/{row}/{name}", **kw)


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
@pytest.mark.parametrize("regime", ["a_square", "b_wide_short", "c_multiblock", "d_smallest"])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_ab_interleaved(device, regime, dtype_name, on, name):
    dtype = ttnn.bfloat16 if dtype_name == "bf16" else ttnn.float32
    _ab(device, f"{regime}/{dtype_name}", B.SHAPES[regime], dtype, on, name)


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
@pytest.mark.parametrize("tile_height", [16, 8, 1])
def test_ab_tile_height(device, tile_height, on, name):
    _ab(device, f"tile_h={tile_height}", B.SHAPES["a_square"], ttnn.bfloat16, on, name, tile_h=tile_height)


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_uint8(device, on, name):
    _ab(device, "uint8", B.SHAPES["a_square"], ttnn.uint8, on, name)


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_crossover(device, on, name):
    """DRAM source -> local HEIGHT shard: W_REGION, the plan the old code served
    with one batched library-helper call. The schedule's primary target."""
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    _ab(device, "crossover", shape, ttnn.bfloat16, on, name, out_mem_config=B._height_shard(shape, cores))


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_crossover_tall(device, on, name):
    shape, cores = [1, 1, 8192, 256], 8
    _ab(device, "crossover_tall", shape, ttnn.bfloat16, on, name, out_mem_config=B._height_shard(shape, cores))


# The two arms above are parametrized, so the knob is a VARIABLE there. The
# ledger's re-runnability check wants a literal forcing arm it can find by
# grepping, and a counterfactual that cannot be re-run is not a counterfactual —
# so B8's two arms are also written out longhand here.
#
# Both hold `split_reader=0`, because the SPLIT READER graduated on this same plan
# later in the round and turns issue-ahead off (it carries its own per-RISC
# transaction ids). With the split live, both B8 arms are the same program and
# measure flat — true, and useless as a counterfactual. Pinning the split off is
# what keeps these arms measuring the thing the ledger's B8 row claims.
def test_bench_lever_read_ahead_on(device):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
        levers=dict(read_ahead=1, split_reader=0),
        label="lever:read_ahead=1/crossover",
    )


def test_bench_lever_read_ahead_off(device):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
        levers=dict(read_ahead=0, split_reader=0),
        label="lever:read_ahead=0/crossover",
    )
