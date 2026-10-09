# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""flat_routed_expert with y_row_major (bf16 row-major y, pack-untilized on the down cores) vs the default bfp8 TILE
y on the same inputs, for the plan families of test_flat_routed_expert_op.py (MiMo 64 experts, K2 reader tails, TP4
2 subgrids, TP2 3 subgrids): every active row must equal the tiled y up to bfp8 rounding, rows outside the active
regions are not written, and both are checked against the quantized-weight reference. Two launches (the second after
a moved arena and different counts: the cache-hit address patch)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.bringup.flat_routed_expert_ttnn.tests.unit.test_flat_routed_expert_op import CASES, _counts, _inputs
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_expert_yrm(device, case):
    tag, H, I, E, m, NG = case
    torch.manual_seed(0)
    weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    gids = [[(4 * e + 1) % NG if E * 4 <= NG else (2 * e + 1) % NG for e in range(E)]]
    op = FlatRoutedExpert(device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, pin=1)
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    pinned = []
    for launch, seed in enumerate((1, 2)):
        counts = _counts(E, m, seed)
        x, offs, x_dev, c_dev, r_dev = _inputs(device, E, H, NG, gids[0], counts)
        y_t = ttnn.to_torch(op(x_dev, c_dev, r_dev)).float()
        y_rm_dev = op(x_dev, c_dev, r_dev, y_row_major=True)
        assert y_rm_dev.layout == ttnn.ROW_MAJOR_LAYOUT and y_rm_dev.dtype == ttnn.bfloat16
        y_r = ttnn.to_torch(y_rm_dev).float()
        worst_d, worst_p = 0.0, 1.0
        for e in range(E):
            c = counts[e]
            if not c:
                continue
            o = offs[e]
            a, b = y_r[o : o + c], y_t[o : o + c]
            # bfp8: 7 mantissa bits with an exponent shared by 16 values -> error <= ~2^-7 of the block max
            blk = b.abs().reshape(c, -1, 16).amax(-1, keepdim=True).expand(-1, -1, 16).reshape(c, -1)
            d = ((a - b).abs() / (blk + 1e-30)).max().item()
            worst_d = max(worst_d, d)
            assert d <= 2**-6, f"{tag} launch {launch} expert {e}: bf16 vs bfp8 y differ by {d} of the block max"
            rows = list(range(0, c, max(1, c // 48)))[:48] + [c - 1]
            xe = x[o : o + c][rows]
            Wg, Wu, Wd = (q(w_) for w_ in weights[0][e])
            ref = (torch.nn.functional.silu(xe @ Wg) * (xe @ Wu)) @ Wd
            ok, pcc = comp_pcc(ref, a[rows], 0.99)
            worst_p = min(worst_p, float(pcc))
            assert ok, (tag, launch, e, c, pcc)
        logger.info(
            f"{tag} launch {launch}: row-major y within {worst_d:.2e} (of block max) of bfp8 y, min PCC {worst_p:.5f}"
        )
        pinned.append(
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
