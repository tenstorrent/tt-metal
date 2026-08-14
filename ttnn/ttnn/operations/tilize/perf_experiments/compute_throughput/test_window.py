# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""How wide may the DEST window legally be?  Correctness-first sweep.

Arm 4 now uses the CORRECTED capacity rule (32-bit input datums halve the DEST
tile count even when DST_ACCUM_MODE is off — the measured uint32 failure).
Arms 8 / 9 push 2x / 4x past it to check whether a TINY tile (tile_h < 32)
occupies less than a full DEST slot; they are probes, so a non-bit-exact result
is DATA, not a test failure.
"""
import pytest
import ttnn
from loguru import logger

from ._harness import VARIANTS, run

SHAPE = [1, 1, 2048, 256]
CORES = 8

CASES = {
    "uint32": dict(dtype=ttnn.uint32),
    "fp32": dict(dtype=ttnn.float32),
    "cast_bf16_to_fp32": dict(dtype=ttnn.bfloat16, out_dtype=ttnn.float32),
    "tile_h8": dict(dtype=ttnn.bfloat16, tile_h=8),
    "tile_h1": dict(dtype=ttnn.bfloat16, tile_h=1),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", [4, 8, 9])
def test_window_probe(device, case, variant):
    kw = dict(CASES[case])
    tile_h = kw.get("tile_h", 32)
    tiles_per_core = (2048 // CORES // tile_h) * (256 // 32)
    ns, exact = run(device, variant, SHAPE, cores=CORES, label=f"window/{case}", **kw)
    logger.info(
        f"WINDOW {case} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact} "
        f"ns_per_tile={ns / tiles_per_core:.2f}"
    )
    if variant == 4:
        assert exact, f"{case}: the corrected DEST window is still not bit-exact"
