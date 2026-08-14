# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS: where is the TRISC floor on the zero-NoC sharded plan?

[1,1,2048,256] bf16, HEIGHT-sharded L1 on 8 cores, SAME spec in and out — both
CBs alias the resident shard, so neither reader nor writer issues a NoC
transaction and the compute pipeline is the wall.  64 tiles/core, 8 blocks of 8.

Arm 0 is the op today.  Arms 1-3 peel the CB handshake off in stages, which is
what separates LLK payload from synchronization.  Arms 4-7 are the wide-DEST
candidate; on THIS case they are byte-identical to arm 0 (bf16 32x32 output takes
the FAST tilize path, which already fills a DEST section per acquire) and are run
only to prove that.
"""
import pytest
import ttnn
from loguru import logger

from .. import _zones
from ._harness import VARIANTS, run

SHAPE = [1, 1, 2048, 256]
CORES = 8
TILES_PER_CORE = (2048 // 8 // 32) * (256 // 32)  # 8 tile-rows x 8 tile-cols


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_focus_arms(device, variant):
    _zones.clear()
    ns, exact = run(device, variant, SHAPE, cores=CORES, dtype=ttnn.bfloat16, label="focus")
    stages, diag = _zones.breakdown()
    compute = {
        # cycles / cores, at the 1000 MHz AICLK the op's other benches assume
        risc: s["cycles"] / max(1, len(s["cores"]))
        for (name, risc), s in stages.items()
        if name == "compute_tilize"
    }
    logger.info(
        f"FOCUS arm={variant}:{VARIANTS[variant][0]} wall_ns={ns} exact={exact} "
        f"ns_per_tile={ns / TILES_PER_CORE if ns else None:.1f} "
        f"compute_zone_ns={ {k: round(v) for k, v in sorted(compute.items())} }"
    )
    assert ns is not None
