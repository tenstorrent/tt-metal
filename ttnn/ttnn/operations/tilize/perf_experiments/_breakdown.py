# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1, Step 1 — the MEASURED per-stage breakdown of tilize.

One fresh-cache run per case (device kernel duration has no warm-up transient),
then the `MaybeDeviceZoneScope` markers the three kernels carry are folded into a
per-stage table by `_zones.py`.

Run:
    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/_breakdown.py

The CASES are the op's distinct kernel paths, one representative each. The two
interleaved-DRAM regimes are included as the ROOFLINE CONTROL: Refinement 3
measured them at or below a pure DRAM->DRAM copy of the same tensor
(87,710 ns @ [1,1,2048,2048], 174,772 @ [1,1,8192,1024]), so they are expected to
show no headroom and any idea aimed at them is expected to be null.
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

import pytest
import ttnn
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))

import _bench_tilize as B  # noqa: E402

from ttnn.operations.tilize.perf_experiments import _zones  # noqa: E402


def _run(device, label, **kw):
    _zones.clear()
    ns = B._measure(device, label=label, **kw)
    try:
        stages, diag = _zones.breakdown()
    except FileNotFoundError:
        logger.warning("no profile_log_device.csv — zone markers unavailable")
        return ns
    freq = device.get_clock_rate_mhz() if hasattr(device, "get_clock_rate_mhz") else 1000.0
    logger.info(f"\n=== BREAKDOWN {label}  whole-op {ns:.0f} ns ===\n" + _zones.report(stages, diag, freq))
    return ns


# --------------------------------------------------------------------------
# (1) DRAM -> local HEIGHT shard crossover. Refinement 6 priced this as the #1
#     remaining opportunity (read half 67% of the wall; the writer issues no NoC
#     at all, so BRISC is idle).
def test_breakdown_crossover(device):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    _run(
        device,
        "crossover/[1,1,2048,256]->H8",
        shape=shape,
        dtype=ttnn.bfloat16,
        out_mem_config=B._height_shard(shape, cores),
    )


# (2) Cross-spec reshard: WIDTH x2 -> HEIGHT x8, a genuine cross-core L1 gather
#     through the R_PAD reader loop. Read half 85%.
def test_breakdown_reshard(device):
    shape, src, dst = B.RESHARD_SHAPE
    _run(
        device,
        "reshard/[1,1,1024,256]W2->H8",
        shape=shape,
        dtype=ttnn.bfloat16,
        in_mem_config=B._width_shard(shape, src),
        out_mem_config=B._height_shard(shape, dst),
    )


# (3) Retile 32 -> 8. Refinement 5 classified this L1-store-bound at ~42 GB/s
#     against the row-major path's ~190.
@pytest.mark.parametrize("in_tile_h,tile_height", [(32, 8), (32, 16)], ids=["32to8", "32to16"])
def test_breakdown_retile(device, in_tile_h, tile_height):
    _run(
        device,
        f"retile/{in_tile_h}to{tile_height}",
        shape=B._RETILE_SHAPE,
        dtype=ttnn.bfloat16,
        tile_h=tile_height,
        in_tile_h=in_tile_h,
    )


# (4) Padded widening cast. Refinement 4 classified this STAMP-bound (360,882 of
#     385,227 ns survives ablating every payload at once).
def test_breakdown_widening_pad(device):
    shape, target = B._OUT_FILL_SHAPE
    _run(
        device,
        "widening_pad/[1,1,1024,2048]->[1,1,2048,2048]",
        shape=shape,
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=target, pad_value=10.2),
    )


# (5) + (6) The interleaved DRAM regimes — ROOFLINE CONTROL, see the module doc.
@pytest.mark.parametrize("regime", ["a_square", "b_wide_short"])
def test_breakdown_interleaved(device, regime):
    _run(device, f"interleaved/{regime}", shape=B.SHAPES[regime], dtype=ttnn.bfloat16)


# (7) Same-spec zero-copy sharded — Refinement 1 called it compute-bound at
#     ~63 ns per 32x32 tile. The only path where the tilize LLK itself could bind.
def test_breakdown_shard_same_spec(device):
    shape, cores = B.SHARDED_SHAPES["e_shard_same_wide"]
    cfg = B._height_shard(shape, cores)
    _run(
        device,
        "shard_same_spec/[1,1,2048,256]H8",
        shape=shape,
        dtype=ttnn.bfloat16,
        in_mem_config=cfg,
        out_mem_config=cfg,
    )
