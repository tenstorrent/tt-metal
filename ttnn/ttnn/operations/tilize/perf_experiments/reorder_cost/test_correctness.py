# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""reorder_cost correctness sweep: kernels from RC_KDIR (default candidate/) on the coalesced
interleaved path (unrolled + general walks, writer share, tile heights, dtypes). torch.equal gate."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # no global torch import under ttnn/ (pre-commit)

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


def _make(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype == ttnn.uint16:
        return torch.randint(0, 60000, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-100000, 100000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape) * 1e30
    return torch.randn(shape).bfloat16()


CASES = [
    # shape, dtype, tile_h, writer_share
    ((1, 1, 16384, 64), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 16384, 32), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 32768, 64), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 3520, 64), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 128, 64), ttnn.bfloat16, 32, 0.0),
    ((2, 3, 2048, 64), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 8192, 1024), ttnn.bfloat16, 32, 0.0),
    ((1, 1, 4096, 64), ttnn.bfloat16, 16, 0.0),
    ((1, 1, 4096, 64), ttnn.bfloat16, 8, 0.0),
    ((1, 1, 4096, 64), ttnn.bfloat16, 4, 0.0),
    ((1, 1, 4096, 64), ttnn.bfloat16, 2, 0.0),
    ((1, 1, 16384, 64), ttnn.bfloat16, 32, 0.5),
    ((1, 1, 16384, 64), ttnn.bfloat16, 32, 0.3),
    ((1, 1, 4096, 64), ttnn.bfloat16, 4, 0.5),
    ((1, 1, 8192, 64), ttnn.float32, 32, 0.0),
    ((1, 1, 8192, 64), ttnn.uint16, 32, 0.0),
    ((1, 1, 8192, 64), ttnn.int32, 32, 0.0),
    ((1, 1, 8192, 64), ttnn.uint8, 32, 0.0),
]


@pytest.mark.parametrize(
    "shape, dtype, tile_h, share", CASES, ids=lambda c: str(c).replace(" ", "").replace("DataType.", "")
)
def test_reorder_correctness(device, monkeypatch, shape, dtype, tile_h, share):
    monkeypatch.setattr(pd, "KERNEL_DIR", Path(__file__).parent / os.environ.get("RC_KDIR", "candidate"))
    monkeypatch.setattr(pd, "WRITER_COPY_SHARE", share)
    torch.manual_seed(0)
    x = _make(dtype, shape)
    t = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = {} if tile_h == 32 else {"tile": ttnn.Tile([tile_h, 32])}
    out = tilize(t, **kw)
    y = ttnn.to_torch(out)
    if x.is_floating_point():
        assert torch.equal(y.to(torch.float64), x.to(torch.float64))
    else:
        assert torch.equal(y.to(torch.int64) & 0xFFFFFFFF, x.to(torch.int64) & 0xFFFFFFFF)
