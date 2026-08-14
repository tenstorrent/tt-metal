# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 — the in-op A/B for the SPLIT READER.

`LEVERS["split_reader"] = 0` restores the single reader, the single input CB, the
writer's drain and the issue-ahead schedule (which the split path turns off), so
the OFF arm is the op exactly as it was before this graduation.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_ab_split_reader.py

Rows: every destination-local plan in the guard set (the split's whole domain),
plus two interleaved-destination rows that must NOT change — the split is
predicated off them because BRISC is issuing the real output writes there, and
Perf 1 measured 0.80x when it was taken anyway.
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
def test_ab_crossover(device, on, name):
    """DRAM source -> resident HEIGHT shard: the SHARED_NOC0 flavor."""
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
        levers=dict(split_reader=on),
        label=f"ab_split/crossover/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_crossover_tall(device, on, name):
    shape, cores = [1, 1, 8192, 256], 8
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
        levers=dict(split_reader=on),
        label=f"ab_split/crossover_tall/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_reshard(device, on, name):
    """Another core's L1 shard -> resident shard: the DEDICATED_DUAL_NOC flavor."""
    shape, src, dst = B.RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        levers=dict(split_reader=on),
        label=f"ab_split/reshard/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_reshard_gated(device, on, name):
    shape, src, dst = B.GATED_RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        levers=dict(split_reader=on),
        label=f"ab_split/reshard_gated/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_padded_local_shard(device, on, name):
    """The R_PAD twin: each reader fills its own blocks' pad region."""
    shape, target, cores = B.PAD_SHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(target, cores),
        pad=dict(output_padded_shape=target, pad_value=3.5),
        levers=dict(split_reader=on),
        label=f"ab_split/padded_local_shard/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
def test_ab_shard_same_spec(device, on, name):
    """Zero-copy BOTH sides — there is no read to split, so this must be flat."""
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    cfg = B._height_shard(shape, cores)
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=cfg,
        out_mem_config=cfg,
        levers=dict(split_reader=on),
        label=f"ab_split/shard_same/{name}",
    )


@pytest.mark.parametrize("on,name", _ARMS, ids=[n for _o, n in _ARMS])
@pytest.mark.parametrize("regime", ["a_square", "b_wide_short"])
def test_ab_interleaved_destination(device, regime, on, name):
    """Interleaved destination: predicated OFF (BRISC is issuing the real writes),
    so both arms must be identical. This row is the check that the predicate holds."""
    B._measure(device, B.SHAPES[regime], ttnn.bfloat16, levers=dict(split_reader=on), label=f"ab_split/{regime}/{name}")
