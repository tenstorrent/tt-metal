# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""DOMAIN: does the wide-DEST regular tilize hold on the op's PRODUCTION plans?

The bake-off above ran the zero-NoC sharded plan so the TRISC was the wall.  Here
the same two arms run the op's real interleaved DRAM->DRAM regimes, where the NoC
competes for the wall — the question is whether the compute win survives, is
masked, or costs anything.  Plus the smallest regime, where a wider DEST window
could plausibly lose.
"""
import pytest
import ttnn
from loguru import logger

from ._harness import VARIANTS, run

# label -> (shape, kwargs)
CASES = {
    # (a) grid-filling square, the op's DRAM-bound regime
    "a_square_h32": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16)),
    "a_square_h16": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, tile_h=16)),
    "a_square_h8": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, tile_h=8)),
    "a_square_h1": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, tile_h=1)),
    "a_square_fp32": ([1, 1, 2048, 2048], dict(dtype=ttnn.float32)),
    "a_square_cast": ([1, 1, 2048, 2048], dict(dtype=ttnn.bfloat16, out_dtype=ttnn.float32)),
    # (b) wide/short — NT_H == 1
    "b_wide_short_h32": ([1, 1, 32, 16384], dict(dtype=ttnn.bfloat16)),
    "b_wide_short_h8": ([1, 1, 32, 16384], dict(dtype=ttnn.bfloat16, tile_h=8)),
    # (c) multi-block per core
    "c_multiblock_h32": ([1, 1, 8192, 1024], dict(dtype=ttnn.bfloat16)),
    "c_multiblock_h8": ([1, 1, 8192, 1024], dict(dtype=ttnn.bfloat16, tile_h=8)),
    # (d) smallest regime — 2 tiles on 2 cores.  A wider DEST window has the most
    #     to lose here (nothing to amortize over).
    "d_smallest_h32": ([1, 1, 32, 64], dict(dtype=ttnn.bfloat16)),
    "d_smallest_h8": ([1, 1, 32, 64], dict(dtype=ttnn.bfloat16, tile_h=8)),
    "d_smallest_h1": ([1, 1, 32, 64], dict(dtype=ttnn.bfloat16, tile_h=1)),
    "d_smallest_uint32": ([1, 1, 32, 64], dict(dtype=ttnn.uint32)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", [0, 4])
def test_domain(device, case, variant):
    shape, kw = CASES[case]
    ns, exact = run(device, variant, shape, sharded=False, label=f"domain/{case}", **kw)
    logger.info(f"DOMAIN {case} arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact}")
    assert exact, f"{case} arm {variant} not bit-exact"
