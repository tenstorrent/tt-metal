# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block factor (WT_CHUNK) at FIXED work, plus repeats of the marginal rows.

WT_CHUNK sweep
--------------
On the aliased same-spec sharded plan the host pins WT_CHUNK to the shard's own
width (tilize_program_descriptor.py:703), so sweeping the shard width at a fixed
tile count per core IS the block-factor sweep: 64 tiles/core in blocks of
1/2/4/8/16/32/64.  Fewer, wider blocks = fewer CB handshakes and fewer
`tilize_block` boundaries; the DEST window is capped independently by the
hardware, so this asks whether the block boundary itself costs anything.

Repeats
-------
The interleaved rows where the arm delta landed inside the ~3% noise band get a
second and third run so the win/flat call is not made on one sample.
"""
import pytest
import ttnn
from loguru import logger

from ._harness import VARIANTS, run

CORES = 8
TILES_PER_CORE = 64

# wt (block factor in tiles) -> shape giving 64 tiles/core on 8 cores at tile_h=32
WT_SHAPES = {
    1: [1, 1, 16384, 32],
    2: [1, 1, 8192, 64],
    4: [1, 1, 4096, 128],
    8: [1, 1, 2048, 256],
    16: [1, 1, 1024, 512],
    32: [1, 1, 512, 1024],
    64: [1, 1, 256, 2048],
}


@pytest.mark.parametrize("wt", list(WT_SHAPES))
@pytest.mark.parametrize("variant", [0, 4])
def test_block_factor_fast(device, wt, variant):
    """bf16 tile_h=32 -> FAST tilize path.  Arms 0 and 4 are identical code here."""
    ns, exact = run(device, variant, WT_SHAPES[wt], cores=CORES, dtype=ttnn.bfloat16, label=f"wt{wt}/fast")
    logger.info(
        f"BLOCKFACTOR fast wt={wt} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} "
        f"exact={exact} ns_per_tile={ns / TILES_PER_CORE:.2f}"
    )
    assert exact


# tile_h=8: same 64 tiles/core, REGULAR path (arms 0 and 4 differ).
WT_SHAPES_H8 = {
    1: [1, 1, 4096, 32],
    2: [1, 1, 2048, 64],
    4: [1, 1, 1024, 128],
    8: [1, 1, 512, 256],
    16: [1, 1, 256, 512],
}


@pytest.mark.parametrize("wt", list(WT_SHAPES_H8))
@pytest.mark.parametrize("variant", [0, 4])
def test_block_factor_regular(device, wt, variant):
    ns, exact = run(
        device, variant, WT_SHAPES_H8[wt], cores=CORES, dtype=ttnn.bfloat16, tile_h=8, label=f"wt{wt}/regular"
    )
    logger.info(
        f"BLOCKFACTOR regular wt={wt} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} "
        f"exact={exact} ns_per_tile={ns / TILES_PER_CORE:.2f}"
    )
    assert exact


REPEAT_CASES = {
    "a_square_h32": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16)),  # identical arms: the NOISE control
    "a_square_h8": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, tile_h=8)),
    "a_square_h1": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, tile_h=1)),
    "d_smallest_h8": ([1, 1, 32, 64], dict(dtype=ttnn.bfloat16, tile_h=8)),
    "d_smallest_uint32": ([1, 1, 32, 64], dict(dtype=ttnn.uint32)),
}


@pytest.mark.parametrize("rep", [1, 2, 3])
@pytest.mark.parametrize("case", list(REPEAT_CASES))
@pytest.mark.parametrize("variant", [0, 4])
def test_repeat(device, case, variant, rep):
    shape, kw = REPEAT_CASES[case]
    ns, exact = run(device, variant, shape, sharded=False, check=False, label=f"repeat/{case}", **kw)
    logger.info(f"REPEAT {case} arm={variant}:{VARIANTS[variant][0]} rep={rep} wall_ns={ns}")
