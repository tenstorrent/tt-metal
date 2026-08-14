# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The REGULAR (non-fast) tilize path — where the DEST window is one tile.

`compute_kernel_lib::tilize` picks `fast_tilize_block` only when the OUTPUT CB
has 32x32 tiles, the input format is bf16/fp32, the output format is not fp32 and
dst_full_sync is off (tilize_helpers.inl:66).  Everything else falls to
`tilize_block`, whose body is one DEST acquire/release per tile on slot 0.

Regimes that land there, all run on the SAME-SPEC height-sharded L1 plan so the
NoC is silent and the TRISC pipeline is the wall:

  * fp32 -> fp32                (fp32 output disables fast tilize)
  * bf16 -> fp32                (cast; fp32 output, so also regular)
  * uint32 / uint16 / int32     (integer formats)
  * uint8                       (8-bit datum path, fp32 DEST forced by the op)
  * tile_h in {16, 8, 4, 2, 1}  (tiny tiles: not 32x32, so never fast)

Arm 4 (`wide_dest`) is the candidate; arm 5 (`raw_regular_ctl`) is the SAME
open-coded loop with the stock `tilize_block`, so arm 5 vs arm 0 isolates the
loop rewrite and arm 4 vs arm 5 isolates the DEST window itself.
"""
import pytest
import ttnn
from loguru import logger

from ._harness import VARIANTS, run

SHAPE = [1, 1, 2048, 256]
CORES = 8

# label -> dict(kwargs for run)
CASES = {
    "fp32": dict(dtype=ttnn.float32),
    "uint32": dict(dtype=ttnn.uint32),
    "uint16": dict(dtype=ttnn.uint16),
    "uint8": dict(dtype=ttnn.uint8),
    "cast_bf16_to_fp32": dict(dtype=ttnn.bfloat16, out_dtype=ttnn.float32),
    "tile_h16": dict(dtype=ttnn.bfloat16, tile_h=16),
    "tile_h8": dict(dtype=ttnn.bfloat16, tile_h=8),
    "tile_h4": dict(dtype=ttnn.bfloat16, tile_h=4),
    "tile_h2": dict(dtype=ttnn.bfloat16, tile_h=2),
    "tile_h1": dict(dtype=ttnn.bfloat16, tile_h=1),
}

ARMS = [0, 5, 4, 6, 7]


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", ARMS)
def test_regular_arms(device, case, variant):
    kw = dict(CASES[case])
    tile_h = kw.get("tile_h", 32)
    tiles_per_core = (2048 // CORES // tile_h) * (256 // 32)
    ns, exact = run(device, variant, SHAPE, cores=CORES, label=f"regular/{case}", **kw)
    logger.info(
        f"REGULAR {case} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact} "
        f"tiles/core={tiles_per_core} ns_per_tile={ns / tiles_per_core:.2f}"
    )
    assert exact, f"{case} arm {variant} not bit-exact"
