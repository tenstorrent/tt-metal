# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1, Step 1b — CUMULATIVE payload ablation on the candidate focus paths.

Stages overlap (NoC reads run in parallel with TRISC compute), so removing one
stage alone under-counts it — the still-running partner fills the gap. These arms
peel the payloads off CUMULATIVELY and end with EVERY payload stubbed at once, so
the "how much of this wall is irreducible sync/overhead?" question is answered by
one run in which nothing is left, never by adding two separate single-peel runs.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_ablation.py
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import pytest
import ttnn

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))
import _bench_tilize as B  # noqa: E402

# read -> read+compute -> read+compute+write (nothing left but the CB handshake)
_PEEL = [
    ({}, "0_full"),
    ({"dm_read": 1}, "1_no_read"),
    ({"dm_read": 1, "compute": 1}, "2_no_read_no_compute"),
    ({"dm_read": 1, "compute": 1, "dm_write": 1}, "3_all_payload_stubbed"),
]


@pytest.mark.parametrize("ablate,name", _PEEL, ids=[n for _a, n in _PEEL])
def test_peel_crossover(device, ablate, name):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
        ablate=ablate,
        label=f"peel_crossover/{name}",
    )


@pytest.mark.parametrize("ablate,name", _PEEL, ids=[n for _a, n in _PEEL])
def test_peel_reshard(device, ablate, name):
    shape, src, dst = B.RESHARD_SHAPE
    B._measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
        ablate=ablate,
        label=f"peel_reshard/{name}",
    )


@pytest.mark.parametrize("ablate,name", _PEEL, ids=[n for _a, n in _PEEL])
def test_peel_retile(device, ablate, name):
    """`dm_read` on the retile path stubs the WHOLE reader — the staging reads AND
    the L1 face permutation — because the reader's ablation branch returns before
    the regime dispatch. That is exactly what makes it the right first peel here:
    the permutation is the payload under test."""
    B._measure(
        device,
        B._RETILE_SHAPE,
        ttnn.bfloat16,
        tile_h=8,
        in_tile_h=32,
        ablate=ablate,
        label=f"peel_retile/{name}",
    )


@pytest.mark.parametrize("ablate,name", _PEEL, ids=[n for _a, n in _PEEL])
def test_peel_wide_short(device, ablate, name):
    """(b) [1,1,32,16384]: ONE block per core, so read/compute/write cannot overlap
    at all. The cumulative peel is what separates "no overlap" from "DRAM-bound"."""
    B._measure(device, B.SHAPES["b_wide_short"], ttnn.bfloat16, ablate=ablate, label=f"peel_b/{name}")
