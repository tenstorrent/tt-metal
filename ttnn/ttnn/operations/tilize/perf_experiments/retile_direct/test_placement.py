# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two axes the focus case pins that could carry a hidden carve-out:

  * BLOCK-FLOAT OUTPUT. `bfloat8_b` is an OUTPUT-only dtype for this op (it is
    not in SUPPORTED["dtype"]), so a retile SOURCE page is always plain elements
    and the direct byte move never has to understand a shared exponent. But a
    bfp8 output IS a cast, i.e. exactly the case the compute-owned datacopy arm
    has to survive — can `copy_tiles` pack block-float?
  * L1 SOURCE. The retile reader reads its source through the TensorAccessor even
    when it is a resident shard (a tiled page cannot back the row-major input CB),
    so an L1 source only changes where the accessor points. Measured, not assumed.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_direct/test_placement.py
"""

import ttnn

from ttnn.operations.tilize.perf_experiments.retile_direct import _harness as H

FOCUS = [1, 1, 1024, 1024]


def test_bfp8_output(device):
    """bf16 -> bfloat8_b, 32 -> 8. A cast whose OUTPUT is block-float."""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8, dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b, oracle="baseline")
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"BFP8-OUT bf16->bfloat8_b 32->8 {FOCUS} (oracle: the op's own output)")


def test_l1_source(device):
    """An L1-INTERLEAVED source instead of DRAM."""
    shape = [1, 1, 512, 512]
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, shape, 32, 8, in_mem_config=ttnn.L1_MEMORY_CONFIG)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"L1-SRC bf16 32->8 {shape}")


def test_padded_retile(device):
    """A retile whose OUTPUT is padded (H not a multiple of the output tile).
    Neither the baseline nor the direct form has any pad handling inside the
    R_RETILE branch, so this asks whether the case is reachable at all and
    whether the direct arms change its answer."""
    shape = [1, 1, 20, 256]  # in_tile_h=1 -> 20 rows; out tile 32 -> pads to 32
    rows = []
    for arm in H.arms_for(1, 32):
        ns, exact = H.run(device, arm, shape, 1, 32, measure=False)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    from loguru import logger

    logger.info("PADDED-RETILE 1->32 [1,1,20,256]: " + str(rows))


def test_multi_image_retile(device):
    """rank-4 with a real batch/channel product: nth_per_img > 1 blocks."""
    shape = [2, 3, 256, 256]
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, shape, 32, 8)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"RANK4 bf16 32->8 {shape}")
