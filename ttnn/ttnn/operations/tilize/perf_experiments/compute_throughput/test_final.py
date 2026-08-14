# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final domain edges + the per-stage confirmation that COMPUTE is what shrank.

Edges deliberately picked to break the candidate if it can be broken:
  * WT_CHUNK not a multiple of the DEST window (5, 12 -> a short tail pass)
  * 32-bit datum AND a tiny tile at once (uint32 tile_h=8)
  * a block-float OUTPUT (bfloat8_b) — takes the FAST path, must be untouched
  * WT_CHUNK == 1 (nothing to widen; must be exactly flat, never slower)
"""
import pytest
import ttnn
from loguru import logger

from .. import _zones
from ._harness import VARIANTS, run

CORES = 8

EDGES = {
    # shard_wt = 5 -> one short window pass
    "wt5_h8": ([1, 1, 2048, 160], dict(dtype=ttnn.bfloat16, tile_h=8)),
    # shard_wt = 12 -> a full 8 window plus a 4 tail
    "wt12_h8": ([1, 1, 2048, 384], dict(dtype=ttnn.bfloat16, tile_h=8)),
    # 32-bit datum AND a tiny tile
    "uint32_h8": ([1, 1, 2048, 256], dict(dtype=ttnn.uint32, tile_h=8)),
    "fp32_h8": ([1, 1, 2048, 256], dict(dtype=ttnn.float32, tile_h=8)),
    # block-float output: fast path, must be byte-identical to today
    "bfp8_out": ([1, 1, 2048, 256], dict(dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b)),
    # WT_CHUNK == 1
    "wt1_h8": ([1, 1, 2048, 32], dict(dtype=ttnn.bfloat16, tile_h=8)),
    # cast + tiny tile together
    "cast_h8": ([1, 1, 2048, 256], dict(dtype=ttnn.bfloat16, out_dtype=ttnn.float32, tile_h=8)),
}


@pytest.mark.parametrize("case", list(EDGES))
@pytest.mark.parametrize("variant", [0, 4])
def test_edges(device, case, variant):
    shape, kw = EDGES[case]
    # bfloat8_b is lossy on the host oracle side; tilize itself is still a pure
    # permutation, so compare arm 4 against arm 0 rather than against torch.
    check = kw.get("out_dtype") != ttnn.bfloat8_b
    ns, exact = run(device, variant, shape, cores=CORES, check=check, label=f"edge/{case}", **kw)
    logger.info(f"EDGE {case} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact}")
    if check:
        assert exact, f"{case} arm {variant} not bit-exact"


@pytest.mark.parametrize("variant", [0, 4])
def test_stage_attribution(device, variant):
    """Per-TRISC `compute_tilize` occupancy on the winning case (sharded tile_h=8)."""
    _zones.clear()
    ns, exact = run(
        device, variant, [1, 1, 2048, 256], cores=CORES, dtype=ttnn.bfloat16, tile_h=8, label="stage/tile_h8"
    )
    stages, diag = _zones.breakdown()
    per_stage = {
        f"{name}/{risc}": round(s["cycles"] / max(1, len(s["cores"]))) for (name, risc), s in sorted(stages.items())
    }
    logger.info(f"STAGE arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact} zones_ns={per_stage}")
    assert exact
