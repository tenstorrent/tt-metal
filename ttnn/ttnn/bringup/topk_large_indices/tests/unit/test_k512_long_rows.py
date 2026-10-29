# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.topk_large_indices k 512 on rows past one fused segment (> 32 chunks, the K' 512 split segmented
body): the k largest values per row, every index in range and pointing at its value. Rows of 33 / 34 / 65 chunks (one-
and two-chunk later segments), tails, a valid_length short of the width, and row counts under and over the core count
(pair split / whole rows). The index top-k shapes (2x2 512 x 27648 / 55296, 8x4 160 x 262144) also run bit-for-bit
repeatable (5 runs, another input between)."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "rows, width, valid",
    [
        (512, 33 * 512, None),
        (512, 34 * 512, None),
        (512, 65 * 512, None),
        (512, 27648, None),
        (512, 55296, None),
        (160, 262144, None),
        (96, 65 * 512, None),
        (160, 40 * 512 + 96, None),
        (512, 55296, 40000),
        (64, 70 * 512 + 32, 70 * 512 + 32),
        (512, 16384, None),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_topk_k512_long_rows(device, rows, width, valid, seed):
    k = 512
    g = torch.Generator().manual_seed(seed * 1000 + rows + width)
    x = torch.randn(1, 1, rows, width, generator=g).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    kw = {} if valid is None else {"valid_length": valid}
    idx = ttnn.to_torch(ttnn.bringup.topk_large_indices(t, k=k, **kw)).to(torch.int64) & 0xFFFFFFFF
    n = width if valid is None else valid
    xs = x.float()[0, 0, :, :n]
    assert int(idx.max()) < n, "index out of range"
    got = torch.gather(xs, -1, idx[0, 0]).sort(-1, descending=True).values
    ref = xs.topk(k, dim=-1).values
    bad = (got != ref).any(-1).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} rows wrong, first {bad[:8].tolist()}"
    assert all(torch.unique(idx[0, 0, r]).numel() == k for r in range(0, rows, max(1, rows // 16))), "duplicate ids"


@pytest.mark.parametrize("rows, width", [(512, 55296), (160, 262144)])
def test_topk_k512_deterministic(device, rows, width):
    from models.demos.common.bringup.testing import determinism

    put = lambda s: ttnn.from_torch(  # noqa: E731
        torch.randn(1, 1, rows, width, generator=torch.Generator().manual_seed(s)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    a, b = put(0), put(1)
    run = lambda t: ttnn.bringup.topk_large_indices(t, k=512, valid_length=width)  # noqa: E731
    determinism.assert_deterministic(lambda: run(a), lambda: run(b), first=run(a), label=f"k512-{rows}x{width}")
