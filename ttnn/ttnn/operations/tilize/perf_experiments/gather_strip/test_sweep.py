# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""DOMAIN SWEEP — the strip form over the op's BLOCKING, not over shapes.

Every cell is (source shard width in tiles = PAGE_TILES) x (block width in tiles
= WT_CHUNK) x dtype x block count. The structural predicate is one line:
`row_bytes % page_bytes == 0` — the block width must be a WHOLE MULTIPLE of the
source shard width. Cells that violate it are here too, to confirm they are
genuinely inexpressible rather than merely untested.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_strip/test_sweep.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.gather_strip import strip_bench as S

# name -> (shape, dtype, src_cores, dst_cores, wt_chunk override)
CASES = {
    # ---- block count ----------------------------------------------------
    "tall_8blk": ([1, 1, 2048, 256], ttnn.bfloat16, 2, 8, None),
    "small_1blk": ([1, 1, 256, 256], ttnn.bfloat16, 2, 8, None),
    # ---- dtype ----------------------------------------------------------
    "fp32": ([1, 1, 1024, 256], ttnn.float32, 2, 8, None),
    # ---- wider W (more tile-columns per block) ---------------------------
    "wide_W512": ([1, 1, 1024, 512], ttnn.bfloat16, 2, 8, None),
    # ---- a ONE-TILE source shard: page = 64 B, 8 slices per block --------
    "src8_1tile_page": ([1, 1, 1024, 256], ttnn.bfloat16, 8, 8, None),
    # ---- source shard = 2 tiles, block = 5 shards (odd multiple) ---------
    "w320_5shards": ([1, 1, 1024, 320], ttnn.bfloat16, 5, 8, None),
    # ---- slices == 1: the block IS one source shard (strip == coalesce) --
    "exact_shard": ([1, 1, 1024, 256], ttnn.bfloat16, 2, 8, 4),
    # ---- INEXPRESSIBLE: block width not a whole multiple of the shard ----
    "frac_2p5_shards": ([1, 1, 1024, 320], ttnn.bfloat16, 5, 8, 5),
    "block_narrower": ([1, 1, 1024, 256], ttnn.bfloat16, 2, 8, 2),
}

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    lines = ["", "=" * 96, "GATHER STRIP — DOMAIN SWEEP (end-to-end device kernel ns)", "=" * 96]
    for case, arms in RESULTS.items():
        row = arms.get("row")
        lines.append(f"  {case}   {arms.pop('_geom', '')}")
        for name, ns in arms.items():
            speed = f"  {row / ns:.2f}x" if row and ns else ""
            lines.append(f"      {name:<12} {ns:>10.0f} ns{speed}")
    logger.info("\n".join(lines) + "\n" + "=" * 96)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arm", ("row", "strip", "strip_fine"))
def test_sweep(device, case, arm):
    shape, dtype, src_cores, dst_cores, wt_chunk = CASES[case]
    p = S.plan(shape, dtype, src_cores, dst_cores, wt_chunk)
    RESULTS.setdefault(case, {})["_geom"] = (
        f"page_tiles={p['page_tiles']} wt_chunk={p['wt_chunk']} slices={p['slices']} "
        f"strip_ok={p['strip_ok']} blocks/core={p['blocks_per_core']}"
    )
    if arm != "row" and not p["strip_ok"]:
        pytest.skip(f"INEXPRESSIBLE: row_bytes={p['row_bytes']} not a multiple of page_bytes={p['page_bytes']}")
    ns = S.run(
        device,
        shape=shape,
        dtype=dtype,
        src_cores=src_cores,
        dst_cores=dst_cores,
        arm=arm,
        wt_chunk=wt_chunk,
        label=f"{case}/{arm}",
    )
    RESULTS[case][arm] = ns
