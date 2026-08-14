# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 guard-set attribution: which graduated lever owns a flagged row?"""
import os, sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
import pytest, ttnn

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))
import _bench_tilize as B


@pytest.mark.parametrize("oc", [1, 0], ids=["overlap_cores=1", "overlap_cores=0"])
@pytest.mark.parametrize("in_tile_h,tile_height", [(1, 32), (32, 8), (32, 16)], ids=["1to32", "32to8", "32to16"])
def test_retile(device, oc, in_tile_h, tile_height):
    B._measure(
        device,
        B._RETILE_SHAPE,
        ttnn.bfloat16,
        tile_h=tile_height,
        in_tile_h=in_tile_h,
        levers=dict(overlap_cores=oc),
        label=f"retile/{in_tile_h}to{tile_height}/oc={oc}",
    )


@pytest.mark.parametrize("oc", [1, 0], ids=["overlap_cores=1", "overlap_cores=0"])
@pytest.mark.parametrize("regime", ["a_square", "b_wide_short", "c_multiblock", "d_smallest"])
def test_interleaved(device, oc, regime):
    B._measure(device, B.SHAPES[regime], ttnn.bfloat16, levers=dict(overlap_cores=oc), label=f"{regime}/oc={oc}")


@pytest.mark.parametrize("oc", [1, 0], ids=["overlap_cores=1", "overlap_cores=0"])
@pytest.mark.parametrize("tile_height", [1, 8])
def test_tile_h(device, oc, tile_height):
    B._measure(
        device,
        B.SHAPES["a_square"],
        ttnn.bfloat16,
        tile_h=tile_height,
        levers=dict(overlap_cores=oc),
        label=f"tile_h={tile_height}/oc={oc}",
    )
