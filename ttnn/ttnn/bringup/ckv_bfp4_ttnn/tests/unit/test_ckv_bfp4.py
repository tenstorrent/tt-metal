# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ckv_bfp4 on one device: the row records equal the host bfloat4_b rule byte for byte, rows outside
[row0, row0 + n) are untouched, the rounded bf16 output holds the records' values, and the op is deterministic."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_spec = importlib.util.spec_from_file_location("ckv_bfp4_reference", Path(__file__).parents[1] / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

FILL = 0xAB


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (n, 1), generator=g).float())
    x[:, :16] *= 50.0  # groups of other scales in the same row
    x[1, 32:48] = 0.0  # an all-zero group
    x[2] = 0.0  # an all-zero row (the masked rows of a chunk)
    return x.bfloat16()


@pytest.mark.parametrize("n, row0, table_rows", [(32, 0, 32), (64, 96, 256), (2560, 128, 2816), (5120, 0, 5120)])
def test_ckv_bfp4_rows(device, n, row0, table_rows):
    x = _inputs(n, n + row0)
    ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(
        torch.full((1, 1, table_rows, 288), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device
    )
    rounded = ttnn.bringup.ckv_bfp4(ckv, rows, row0, rounded=True)
    got = ttnn.to_torch(rows).view(torch.uint8).reshape(table_rows, 288)
    exp = ref.encode_rows(x)
    bad = (got[row0 : row0 + n] != exp).any(-1).nonzero().flatten()
    assert not len(bad), f"{len(bad)} rows differ, first {bad[:5].tolist()}"
    assert (got[:row0] == FILL).all() and (got[row0 + n :] == FILL).all(), "rows outside [row0, row0 + n) written"
    r = ttnn.to_torch(rounded).reshape(n, 512)
    assert torch.equal(r.float(), ref.decode_rows(exp)), "rounded bf16 != the records' values"

    other = ttnn.from_torch(
        _inputs(n, n + row0 + 1).reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    def run(inp):
        out = ttnn.bringup.ckv_bfp4(inp, rows, row0, rounded=True)
        return [rows, out]

    determinism.assert_deterministic(lambda: run(ckv), lambda: run(other), first=[rows, rounded], label=f"n{n}")


def test_ckv_bfp4_without_rounded_output(device):
    x = _inputs(64, 7)
    ckv = ttnn.from_torch(x.reshape(1, 1, 64, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(torch.zeros(1, 1, 64, 288, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
    assert ttnn.bringup.ckv_bfp4(ckv, rows) is None
    assert torch.equal(ttnn.to_torch(rows).view(torch.uint8).reshape(64, 288), ref.encode_rows(x))
