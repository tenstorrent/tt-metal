# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for every bake-off arm — run this BEFORE the perf file.

tilize is a permutation, so the bar is BIT-EXACT (`torch.equal`, no PCC): a
faster wrong answer is disqualified, and a rounding-tolerant check could not see
a mis-addressed face row anyway.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_permute/test_correctness.py
"""

import pytest
from loguru import logger

from ttnn.operations.tilize.perf_experiments.retile_permute import _harness as H

# Small, so a broken arm fails (or hangs) cheaply. Multi-core still, and >1 block
# per core, so the staging cache and the block loop are both exercised.
SHAPE = [1, 1, 512, 256]

# (in_tile_h, tile_h): splits, merges, and the identity-height case.
CASES = [(32, 8), (32, 16), (1, 32), (8, 32), (32, 4), (16, 32), (2, 32), (32, 2)]


@pytest.mark.parametrize("in_tile_h,tile_h", CASES, ids=[f"{a}to{b}" for a, b in CASES])
def test_arms_bit_exact(device, in_tile_h, tile_h):
    bad = []
    for variant in H.arms_for(in_tile_h):
        try:
            _, exact = H.run(device, variant, SHAPE, in_tile_h, tile_h, measure=False)
        except Exception as exc:  # a geometry the OP itself refuses is not an arm result
            if variant == 0:
                pytest.skip(f"op refuses {in_tile_h}->{tile_h}: {exc}")
            raise
        if not exact:
            bad.append(f"{variant}:{H.VARIANTS[variant][0]}")
    logger.info(f"CORRECTNESS {in_tile_h}->{tile_h}: bad={bad or 'none'}")
    assert not bad, f"arms not bit-exact on {in_tile_h}->{tile_h}: {bad}"
