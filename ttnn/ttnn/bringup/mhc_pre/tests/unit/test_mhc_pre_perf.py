# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe for mhc_pre: one call per shape (run under run_safe_pytest.sh --profile).

Correctness is only smoke-checked (finite outputs); the acceptance suite owns correctness.
"""

import pytest
import torch
import ttnn

from ttnn.bringup.mhc_pre import mhc_pre

PERF_SHAPES = [
    (1, 1, 640, 4 * 7168),
    (1, 1, 640, 4 * 1792),
    (1, 1, 1280, 4 * 4096),
    (1, 1, 4096, 4 * 1792),
]


@pytest.mark.parametrize("x_dtype", [ttnn.float32, ttnn.bfloat16], ids=["xf32", "xbf16"])
@pytest.mark.parametrize("x_shape", PERF_SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_perf(device, x_shape, x_dtype):
    torch.manual_seed(0)
    nc = x_shape[-1]
    x = torch.randn(x_shape, dtype=torch.float32)
    w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
    b = torch.randn((1, 24), dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y, post, comb = mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
    for t in (y, post, comb):
        assert torch.isfinite(ttnn.to_torch(t)).all()
